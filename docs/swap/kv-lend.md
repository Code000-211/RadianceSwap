# Modellübergreifende KV-Leihe (#17)

Stand: Teil 1 (#36) Analyse und Planner-Schicht, Teil 2 die Anbindung an die Engine (unten ab „Teil 2“).
Begriffe: `GLOSSARY.md` (KV-Partition, KV-Leihe). Vorgänger: `docs/swap/slot-release.md` (#15),
`docs/swap/elastic-slab.md` (#35).

Der Abschnitt „Was heute gilt“ und die Tabelle beschreiben den Stand **vor** Teil 2; was sich geändert hat,
steht in „Teil 2“.

## Was heute gilt

**Der KV-Pool ist elastisch, aber nur im Adressraum.** `Pools::configure` legt `vram_kv` mit
`Pool::init_elastic` an (VMem: Adressbereich reservieren, Seiten per `commit` binden). Der Bereich ist
der Plan-Anteil `vb.kv`. `KVManager::carve_into` bindet aber den ganzen Carve beim Start
(`pool_->commit(0, used)`, `core/mem/kv.cpp`) und nichts gibt ihn je wieder frei. Der KV-Pool eines
Models wächst also nie über seinen Carve und schrumpft nie. `n_blocks` pro Gruppe steht fest, Block-IDs und
Block-Tabellen sind darauf ausgelegt.

**Der KV-Loan ist rein logisch und innerhalb einer Engine.** `Engine::loan_tick` fragt
`KVManager::spare_tokens()` (Abstand vom höchsten gehaltenen Block bis zur Carve-Spitze, minus Puffer),
setzt die Marke `set_loan()` und sagt jedem `Mover` per `set_loan(kLendKV, …)` das Angebot. Der Mover
legt Expert-Units in die Slots, die `lend_from(kLendKV, strips)` aus den **Adressen des eigenen KV-Carve**
geschnitten hat (`engine_bringup.cpp`, "the loan"). Es gibt keine Allokation und kein Mapping. Braucht
die Cache-Seite die Blöcke zurück, ruft `KVManager::set_reclaim` (in `start()` installiert) auf dem
Scheduler-Thread `Mover::reclaim(kLendKV, …)` auf allen Ranks **derselben Engine**: Units fallen auf ihre
Host-Kopie zurück, beide Streams werden gedraint. Laufende Sequenzen werden nie verdrängt: ein gehaltener
Block (Sequenz oder Prefix-Cache) wird nie verliehen.

**Der Expert-Slab liegt im Gewichte-Pool** (eine `rad_dev_alloc`, Bump-Allokator). Er ist der einzige
Ort, an dem ein Model Experts hält, ausgenommen die geliehenen KV-Adressen. Seine Obergrenze ist der
feste Anteil des Models (`manifest_headroom_mib`, `core/main.cpp`).

Daraus folgt, was die modellübergreifende Leihe physisch braucht:

| Richtung | Was passieren muss | Heute möglich? |
|---|---|---|
| B holt KV-Platz, A's Slab füllt den Anteil | Bytes aus A's Slab wandern in B's KV-Pool | **Nein.** A's Slot ist Teil von A's Gewichte-Pool und nicht entmappbar (#35). B's Carve ist fest und kann nicht wachsen. |
| Idle KV von A an den Slab von B | B's `Mover` schneidet Slots aus A's KV-Adressen (`lend_from`) | **Nein, nicht korrekt.** Die Adressen sind prozessweit gültig, aber ein `Mover` darf nur bewegt werden, solange sein Rank-Thread geparkt ist, und A's Recall-Hook läuft auf A's Scheduler-Thread. Es fehlt ein Protokoll zwischen den Engines (wer parkt wen, in welcher Reihenfolge, wer hält die Lebensdauer von A's Pool, wenn A deaktiviert wird). |
| Idle KV von A an den Slab von A | wie heute | ja, unverändert |

## Kleinste korrekte Form

Eine Partition holt sich Platz **nie von einer anderen Partition**, sondern aus dem **Expert-Pool der
Gruppe**: Die Bytes, die B's Partition über ihren Carve hinaus braucht, werden vom Expert-Budget
abgezogen, bevor der GroupPlanner es nach Wärme verteilt. Wer abgibt, entscheidet damit die vorhandene
Rangfolge, kein zweiter Algorithmus: die kältesten Units nach Wert (`Wärme * Bytes / Slot-Bytes`) eines
beliebigen Models, B's eigene eingeschlossen, wobei jedes Model sein Minimum (`max(expert_floor,
min_share * Gleichanteil)`) behält. Dass das auch A's idle geliehene KV-Blöcke trifft, ist kein
Sonderfall: ein Expert auf einem geliehenen Block gehört zu A's Slab, und ihn zu verdrängen *ist* die
Rückforderung.

Das steht in `core/place/group_planner.{h,cpp}`:

- `KvPartition` (je Rank, Bytes): `carve`, `held`, `want`, `cap`, `growable`. `RankFacts::kv`,
  `GroupPlanner::update_kv`.
- `RankBudget::kv_grow`: wie viele Bytes die Partition über ihren Carve hinaus bekommt.
- Anspruch einer Partition: `clamp(want, held, max(cap, held)) - carve`, gedeckelt auf `growable`.
- Ansprüche werden nur aus dem bedient, was die Minima der Models übrig lassen. Reicht es nicht, teilen
  sie es im Verhältnis ihrer Ansprüche. Der Rest des Pools geht wie bisher an die Experts. Experts plus
  `kv_grow` übersteigen den Pool nie, VRAM wird nicht doppelt verplant.
- **Laufende Sequenzen werden nie verdrängt**: `held` ist die Untergrenze von `want`, und keine Ausgabe
  nennt Bytes, die eine Partition abgeben soll. Der Planner kann Partitionen nur vergrößern.
- **Single-Engine unverändert**: eine einzelne Engine (kein pending) bekommt wie bisher ihr Angebot, ein
  KV-Bericht ändert nichts.
- Endet der Bedarf (`want` fällt), verschwindet der Anspruch bei der nächsten Neuberechnung und die
  Experts bekommen ihr Budget zurück.

## Teil 2: die Engine wächst (#17)

Die zwei Lücken aus Teil 1 sind zu: der Expert-Slab liegt auf elastischem Speicher (#35), und der KV-Pool
eines Models kann über seinen Carve hinaus wachsen. Die Planner-Schicht (`growable`, Anspruch `clamp(want,
held, cap) − carve`) hängt jetzt an der Engine.

### Der KV-Pool wächst in Stücken auf frischem Adressraum

`KVManager` (`core/mem/kv.{h,cpp}`, Abschnitt „growth“) trägt dieselbe Regel wie der Slab (`elastic-slab.md`):
**eine Adresse wird nur einmal gemappt.** `hsa_amd_vmem_map` auf eine schon gemappte und wieder entmappte Adresse
wirkt nicht; zwei Adressen läsen dieselbe Seite.

- **Layout.** Die Pools sind layer-major (`base + layer * layer_stride`), der Blockindex liegt innerhalb des
  Layers. Wachsen darf also keine Adresse bewegen, und die Bindungen der Kernel (Basis, Stride, Form) bleiben
  fest. Eine wachsende Gruppe (die lendable, FULL-Gruppen) bekommt deshalb pro Layer einen Streifen, der so tief
  ist wie alle Block-IDs, die je gebraucht werden (`KVGroupPlan::va_blocks`), aus dem Adressraum hinter dem
  Budget (`Pool::va_arena`, `Pools::configure(..., kv_va_extra)`, 4 TiB je Rank, nur virtuell). Gebackt wird nur der Carve
  (`carve_blocks`). Die Budget-Reservierung der Gruppe bleibt nur als Buchführung; nichts davon wird gebackt.
- **Ein Stück** (`back_chunk`) backt die nächsten IDs in jedem Layer jeder wachsenden Gruppe, in allen Gruppen
  gleich viele Token (sie laufen gemeinsam voll). Es beginnt und endet auf einer Commit-Granule jedes Streifens
  (`chunk_align`), und ist etwa 128 MiB groß wie die Slab-Stücke. Ein Stück, das zurückgegeben wird
  (`retire_chunk`, nur das oberste und nur wenn jeder Block darin frei ist), wird nie wieder gemappt. Seine IDs
  sind tot (`refcount −1`, nicht in der Free-Liste) und das nächste Stück nimmt die IDs danach. Die Zahl der
  Stücke ist durch den Adressraum begrenzt (4 TiB bei 128-MiB-Stücken: rund 32 000 Backungen je Rank).
- **Zusammenspiel mit der Leihe.** Die Leihe (`set_loan`) verleiht die obersten Blöcke des *Carve*
  (`LoanStrip` endet am Carve, nicht am Streifen). Beides schließt sich aus: mit Stück gibt es nichts zu leihen
  (`spare_tokens` = 0, `set_loan` > 0 wird abgelehnt), und ein Stück wird nur gebackt, wenn nichts verliehen ist.
  Das ist keine Einschränkung: ein Cache, der wächst, hat keine idle Blöcke.
- **Alle Ranks** backen dasselbe Stück (IDs sind rankweit gleich), danach bietet Rank 0 die Blöcke an
  (`publish_chunk`). Scheitert ein Rank (die Karte lehnt ab), geben die anderen es zurück (`abandon_chunk`); die
  versuchten IDs sind verbraucht.
- Wächst nur, wenn alle Gruppen mitwachsen können: keine Zero-, Streaming- oder Runtime-Window-Gruppe, jede
  Fix-Window-Gruppe am Pool-Deckel, Layer-Bytes und Granule passen zusammen und alle Ranks haben dieselbe
  Stückgröße. Sonst bleibt der Cache fest (`growth_enabled()` = false).
- **Wachsen ist nur Kapazität.** `can_ever_fit`, `ensure` und die Zähler (`total_blocks`, `held_blocks`) rechnen
  mit den Blöcken, die gerade Seiten haben; die tote ID-Strecke zählt nicht. Eine Anfrage, die größer ist als die
  heutige Kapazität, wird wie bisher abgewiesen (ein Wachsen für eine einzelne Sequenz würde das Verdrängen von
  Experts an einen Fehlversuch hängen).

### Wie die Engine es treibt (`Engine::group_tick`, `kv_growth_tick`)

Alle 100 ms, zwischen Steps und unter dem Scheduler-Lock (wie die Leihe):

1. **Melden.** `KvPartition{carve = Σ pool_bytes_carved, held = Σ pool_bytes_used, want = held + buffer_bytes(),
   cap = ceiling(max_ctx), growable = (Stücke + freie Stücke) * Stückgröße, grown = Stücke * Stückgröße}`. `want` und
   `cap` über dem Carve in ganzen Stücken aufgerundet (ein Anspruch von ein paar Blöcken soll nicht auf 0 Stücke
   abrunden). Gemeldet wird, wenn sich ein Wert um ein Stück bewegt.
2. **Anwenden.** Ziel = `kv_grow / Stückgröße`, das Kleinste über die Ranks. Über den Stücken: eines pro Tick
   backen, sofern keine Leihe offen ist und die Karte danach `gpu_headroom_mib` frei behält (wie
   `MoverConfig::grow_floor`). Unter den Stücken: nach 5 s das oberste zurückgeben, sobald jeder Block darin frei
   ist (Streams werden vorher gedraint). Ein Block, den eine Sequenz oder der Prefix-Cache hält, hält das Stück.

### Der Planner: nie VRAM doppelt verplanen (`group_gate`)

`group_allocate` schneidet den Expert-Pool um den Anspruch, wie in Teil 1 (die kältesten Units eines beliebigen
Models, jedes behält sein Minimum). Neu in `group_gate`: Ein Anspruch wird **erst bedient, wenn die Bytes physisch
frei sind** (`Pool − Σ gehalten`), genau wie der Wachstumsanspruch eines elastischen Slabs:

- Der Geber bekommt sein kleineres Ziel sofort (`Target < held`), schrumpft Stück für Stück und meldet `held`.
- Dem KV-Anspruch wird nur gewährt, was frei ist, und **vor** dem Wachstum elastischer Slabs. Ein gewährter, noch
  nicht gebackter Anspruch (`prev`) bleibt reserviert, solange er gewollt wird: zwei Nehmende bekommen nie dieselben
  freien Bytes.
- `KvPartition::grown` sind die Bytes, hinter denen das Pool Seiten hat. Sie zählen als gehalten, bis die Engine sie
  zurückgegeben hat und es meldet. Fällt der Anspruch unter `grown`, bekommt der Geber die Bytes **nicht** schon
  jetzt; `kv_grow` unter `grown` ist das Signal an die Engine, Stücke zurückzugeben.
- Laufende Sequenzen: `held` ist die Untergrenze von `want`; kein Ausgang nennt Bytes, die eine Partition abgeben
  soll. Die Rückgabe trifft nur freie Blöcke.

`/radiance/models` meldet je Rank `kv_growth{grown_bytes, target_bytes}`, `/metrics`
`radiance:model_kv_grown_bytes` und `radiance:model_kv_grow_target_bytes`.

### Die Regel für die KV-Obergrenze eines Models (neu)

Bisher begrenzte der feste Anteil eines Models (`manifest_headroom_mib`, `core/main.cpp`) seinen KV-Cache: mehr
als der Carve war nicht möglich. Jetzt **legt der Anteil nur noch den Start-Carve fest**. Die Obergrenze des Cache
ist seine eigene (`ceiling(max_ctx)`, `--max-model-len` × `--max-num-seqs`), und alles über dem Carve kommt aus dem
Expert-Pool der Gruppe, solange ihn ein Model physisch zurückgibt und die Minima erhalten bleiben (`min_share`,
`expert_floor`). Gilt nur in der Manifest-Gruppe mit mehr als einem Model und VMem; ein einzelnes Model hat keinen
Arena-Adressraum und ändert sich nicht. `RADIANCE_ELASTIC_SLAB=off` schaltet es wie den Slab ab.

### Was bleibt

- **Idle KV von A an den Slab von B** (`lend_from` über Engine-Grenzen) ist weiter nicht gebaut: dafür fehlt das
  Protokoll zwischen den Engines (Tabelle oben). Die Leihe von A's idle Blöcken an A's eigenen Slab läuft unverändert.
  Dass B Platz bekommt, geschieht über den Expert-Pool: A's Slab (das Ziel der Leihe) schrumpft und seine Bytes
  gehen an B's KV.
- **Latenz.** Die Kette Anspruch → Geber schrumpft (Stück für Stück) → Meldung → Backen dauert Sekunden. Wächst B's Last schneller, lehnt `ensure` ab und der
  Scheduler präemptiert wie bei jedem vollen Pool (Neuberechnung, kein Verlust von KV-Daten einer laufenden
  Sequenz, aber Rechenzeit); `buffer_bytes()` plus die Aufrundung auf ein Stück halten den Anspruch dem Bedarf voraus.
- Das oberste Stück bleibt, solange es einen Block hält, auch einen des Prefix-Cache. Darunter liegende freie Stücke
  werden erst frei, wenn es geht.
- Ein Model, das als einziges in der Gruppe aktiv ist, bekommt wie bisher sein Angebot (Single-Engine unverändert),
  keinen KV-Anspruch. Der Anspruch entsteht erst mit einem zweiten Mitglied.
- Die Anzahl der Stücke je Rank ist endlich (Arena). Ist sie verbraucht, wächst der Cache nicht mehr
  (`chunks_left()` = 0, `growable` fällt auf `grown`).

## Tests

- `tests/group_planner_test.cpp`, Abschnitt „KV partitions“ (Teil 1: Anspruch, Minima, `held`, `cap`, `growable`,
  anteilig, Single-Engine, World Size 2) und Abschnitt „the cache grows for real“ (Teil 2: Gewährung erst nach
  Freigabe, `grown` zählt als gehalten, ein gewährter Anspruch bleibt reserviert, Entscheidung je Rank).
- `tests/mem_test.cpp`, Abschnitt „the cache grows“: ohne Arena fest, nur der Carve gebackt, ein Stück fügt
  Blöcke in allen Layern hinzu (Host-VMem über `mmap` ist die Fake-Device-Schicht), Sequenz läuft über den Carve
  ins Stück, Rückgabe nur bei freiem Stück ohne Verlust, IDs/Adressen nie wiederverwendet, Leihe und Wachstum
  schließen sich aus, Follower endet mit denselben IDs, ein Zufallslauf (400 Schritte, Sequenzen, Wachsen,
  Zurückgeben, Leihe) mit Buchführung jedes Blocks, und das Protokoll der Engine gegen den echten Planner
  (`a_partition_grows_out_of_the_slab_another_model_gave_back_and_returns_it`: Summe aus A's Slab und B's Stücken
  übersteigt den Pool nie).
- GPU-Abnahme (`kvgrow.py`, `m_kv.json`, `docs/TOOLS.md`) steht aus; sie misst auf der Karte, was die Host-Tests
  nicht können: B bekommt KV, A's Slab sinkt, `lpstress`-artige Logprobs ohne nicht endliche Logits, laufende
  Sequenzen ungestört.
