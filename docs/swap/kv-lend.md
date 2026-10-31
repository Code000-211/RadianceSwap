# Modellübergreifende KV-Leihe (#17)

Stand: Analyse, Planner-Schicht und die verbleibende Lücke. Begriffe: `GLOSSARY.md` (KV-Partition,
KV-Leihe). Vorgänger: `docs/swap/slot-release.md` (#15). Voraussetzung für den physischen Teil: #35.

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

## Warum nichts davon in der Engine hängt

Ein Anspruch wird nur bedient, soweit der Pool die Bytes auch binden kann (`growable`). Heute kann
das kein Pool: B's Carve ist fest, und A's abgegebener Slot bleibt in A's Gewichte-Pool. Würde die
Engine trotzdem Ansprüche melden, verdrängte der Planner Experts für Bytes, die B nicht nutzen kann, und
kostete nur Decode-Zeit, also eine Halblösung, die VRAM auf dem Papier zweimal vergibt. Die Engine
meldet deshalb keine Partition (`growable` = 0 ist der Standard), `Engine::group_tick` und alle
Produktionspfade sind unverändert.

## Verbleibende Lücke (für #35 und eine Folgearbeit)

1. **Expert-Slab auf elastischem Speicher (#35).** Verdrängte Slots müssen physisch freiwerden
   (`Pool::decommit` der obersten Segmente des Gewichte-Pools), damit die Karte die Bytes frei hat.
2. **KV-Carve, der wachsen kann.** `KVManager` braucht ein Regrow nach dem Start: Adressbereich
   (`init_elastic` reserviert nur `vb.kv`, größer reservieren ist billig), mehr Blöcke pro Gruppe
   (`n_blocks`, Free-Liste, `held_hist_`, Block-Tabelle auf allen Ranks), `commit` der neuen Seiten an
   Step-Grenzen. Danach meldet die Engine in `group_tick` `KvPartition{carve=pool_bytes_carved,
   held=pool_bytes_used, want=held+buffer_bytes(), cap=ceiling, growable=<committbar>}` pro Rank und
   setzt `kv_grow` um (Rank 0 führt, Follower folgen wie bei `follow()`).
3. **Idle KV an den Slab eines anderen Models.** Ein Engine-übergreifendes Protokoll für
   `lend_from`/`reclaim` (siehe Tabelle oben). Planner-seitig braucht es dann eine Verteilung der idle
   Bytes nach Wert auf die Slabs aller Models; sie ist noch nicht gebaut, weil sie ohne das Protokoll
   nichts bindet.
4. Die Rückgabe (`kv_grow` sinkt) muss in der Engine als `decommit` der obersten freien Blöcke
   ausgeführt werden, nie unter `held`.

Bis dahin gilt: jeder Anteil bleibt fest, die Leihe läuft nur innerhalb einer Engine, und die
modellübergreifende Rückforderung ist als geprüfte Entscheidungsschicht vorhanden, aber nicht
angeschlossen.

## Tests

`tests/group_planner_test.cpp`, Abschnitt "KV partitions": B bekommt KV, obwohl A's Slab den Anteil
füllt; die Rückforderung trifft das kältere Model (auch das eigene); Minima und Expert-Floor bleiben;
laufende Sequenzen (`held`) werden nie angetastet; Deckel durch `cap` und `growable` (0: kein Anspruch,
keine Verdrängung); Ansprüche teilen sich anteilig; Single-Engine unverändert; Neuberechnung bei
`update_kv`; World Size 2 pro Rank.
