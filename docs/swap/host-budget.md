# Host-Budget und LRU zum kalten Model (#13)

`cache.host_budget_mib` im Manifest (0 = aus den bestehenden Host-/cgroup-Grenzen ableiten) ist ein
gemeinsames Budget für das Host-RAM, das der Prozess für Models belegt. Geführt wird es im
`PreparedWeightStore` (`core/model/weight_store.{h,cpp}`), dem Ledger des Prozesses:

| Art (`kind`) | Inhalt | Wer bucht |
|---|---|---|
| `store` | Store-Einträge und laufende Captures, vorbereiteter **und aktiver** Models | `reserve`/`commit` |
| `pools` | Host-Pool (inkl. Embedding-/Tower-Pins und Drafter-Experten), Host-Tier des Prefix-Caches, Aktivierungs-Arenen, Checkpoint-Store, n-gram-RAM-Kopie eines **aktiven** Models | `Engine::plan`, `configure_kv` (`claim_host`) |
| `staging` | gepinnter Ring der Load-Worker (8 × 4 MiB je Worker) und das Relayout-Fenster (höchstens 4 GiB), nur für die Dauer des Loads | `Engine::load_weights` |

Pools und Staging eines Models enden mit seinem `~Engine` (`deactivate`). Nicht gebucht (Grenze,
siehe unten): KV-Spiegel (`kvstream`), Pinned-Puffer des Schedulers und Mapped-/File-Tier.

## Verdrängung

- **Capture** (Füllen beim ersten Load): braucht `store + pools + staging + Kopie ≤ Budget`. Fehlt
  Platz, wird das vorbereitete Model verdrängt, das am längsten ungenutzt ist (LRU nach der letzten
  Deaktivierung bzw. dem letzten Load), und wird **kalt** (`/radiance/models`: `state: cold`,
  `registry` Zustand `Cold`). Ein aktives Model wird dafür nie angetastet. Ist auch nach dem Verdrängen
  aller vorbereiteten Models kein Platz (`make_room`), füllt das aktivierende Model den Store nicht
  und ist nach der Deaktivierung kalt. Der Platz wird vor dem ersten Gewicht geprüft; was ohnehin
  nicht passt, verdrängt nichts.
- **Pflichtbuchung** (Pools/Staging einer Aktivierung): wird immer gebucht. Übersteigt sie das
  Budget, werden zuerst vorbereitete Models verdrängt, dann die **Store-Kopie aktiver Models**
  (das Model bleibt aktiv, ist nach Deaktivierung kalt), zuletzt die eigene Kopie des Aktivierenden.
  Pools aktiver Models werden nie verdrängt: bleibt die Aktivierung trotzdem über dem Budget,
  wird sie abgelehnt (siehe „Host-Pools vor der Allokation“, #46).
- Ein Model, dessen Load läuft (`begin` … `commit`/`abort`), wird nie verdrängt (der Load hält Zeiger
  in seine Einträge).
- **Ohne Budget** (0) entscheiden die bestehenden Checks (`lockable_room`: MemAvailable,
  cgroup-Ebenen v1/v2, bei gelockten Kopien RLIMIT_MEMLOCK, `--host-headroom-mib`). Sie laufen über
  `Engine::host_room`: lehnt `lockable_room` ab und hält der Store Kopien, wird die am längsten
  ungenutzte verdrängt und neu geprüft. Auch mit Budget gelten diese Checks zusätzlich. Pinned
  Speicher, den die Karte liest, zählt (wie bisher) nicht gegen RLIMIT_MEMLOCK.
- Das Verdrängen gibt den Speicher sofort zurück: Store-Puffer sind anonyme `mmap`s (`munmap`), kein
  malloc-Arena, die ihn behalten könnte.

## Host-Pools vor der Allokation (#46)

Bei der GPU-Messung zu #18 (PR #43) endete MoE `a` (138 GiB Host-RAM) + streamendes MoE `b` mit Exit 137:
`b` vergrößerte seinen Host-Pool (gestreamte Statik, Experts ohne Kartenplatz), und nur das Ledger
wurde gebucht, nichts geprüft. Jetzt hat `Engine::plan` am Ende der Pool-Planung (nach Drafter- und
Streaming-Wachstum, vor jeder Allokation) ein letztes Wort, `Engine::admit_host_pools`:

1. **Host-Checks** (`host_room`, auch ohne `cache.host_budget_mib`): die gesamte Summe, die `plan` für diese
   Aktivierung bucht (Host-Pool aller Ränge inkl. Wachstum, Embedding-/Tower-Pins, Host-Tier des
   Prefix-Caches, Arenen, n-gram-RAM-Kopie), muss gegen MemAvailable, jede Memory-cgroup und
   `--host-headroom-mib` passen. Passt sie nicht, werden zuerst Store-Kopien verdrängt (vorbereitete Models,
   dann Kopien aktiver), und es wird neu geprüft.
2. **Budget**: ist `cache.host_budget_mib` gesetzt und die Buchung bleibt auch nach dem Verdrängen jeder
   verdrängbaren Kopie darüber, wird abgelehnt.

Abgelehnt heißt: `plan` liefert `RAD_E_NOHOST` („not enough host memory“), die Engine wird mit allen Buchungen
zerstört, die Meldung im Log nennt Grenze und Größe. Der Router behandelt das **nicht** als VRAM-Mangel
(`RAD_E_NOMEM`): kein anderes Model wird dafür deaktiviert; die Anfrage bekommt 503
(`model_activation_failed`, `/radiance/models`: `state: failed`, `error`), mit Backoff wie jede
gescheiterte Aktivierung. Ein laufendes Model `a` behält seine Pools, seine Anfragen laufen weiter; höchstens
seine Store-Kopie (nicht der Pool) wurde verdrängt, es ist dann nach der Deaktivierung kalt.

Nur die Pools dieser Aktivierung laufen so strikt durch das Budget; Staging, Checkpoint-Store und die
einzelnen Pins in `resolve_table_placement` buchen weiter wie oben (Warnung, kein Abbruch).

Tests (Host): `lifecycle_test` (`a_host_pool_that_does_not_fit_refuses_the_activation_and_leaves_the_active_model`:
simuliertes MemAvailable/cgroup über `--host-headroom-mib`, simuliertes Budget über `set_budget`; ein zweiter
Engine neben dem aktiven scheitert mit `RAD_E_NOHOST`, die Pools des aktiven bleiben, passt es ins Budget, wird es
aktiv; `a_failed_start_releases_every_partial_allocation_at_every_step`: ein Host-Pool größer als die Maschine),
`model_deactivation_test` (`an_activation_whose_host_pools_do_not_fit_is_refused_and_the_active_model_stays`: 503,
`a` bleibt aktiv, keine Druck-Deaktivierung). GPU-Nachweis (`a` MoE aktiv + `b` MoE streamend, kein Exit 137):
`scripts/swap-acceptance/host_budget_stream.py`.

GPU-Nachweis (2× R9700, 170 GiB RAM, siehe `gpu-acceptance.md` §13): `a` = MoE mit 73 GiB Pool + 67 GiB Store-Kopie,
`b` = dasselbe MoE, streamt 56 %. Ohne Budget: `b` wird zugelassen (200), nachdem die Store-Kopie des aktiven `a`
verdrängt wurde; Spitze 154,6 GiB belegt, MemAvailable nie unter 16 GiB. Mit Budget = Pools von `a` + 256 MiB:
503 `not enough host memory`, `b` `failed`, `a` antwortet unverändert. Kein Exit 137.

## Page-Cache ist kein RAM-Backing

Nur anonymer Speicher des Prozesses wird gebucht. Mapped-Tier, File-Tier und Container-Reads über den
Page-Cache haben keinen Store-Eintrag; `cgroup_room` zieht Page-Cache-Seiten vom Verbrauch ab, der
Kernel gibt sie jederzeit zurück (Test `hostmem_test`).

## Status und Metriken

`/radiance/models` (`host_memory`): `budget_bytes` (0 = abgeleitet), `used_bytes`, `store_bytes`,
`pool_bytes`, `staging_bytes`, `evicted_prepared`, `evicted_active_copies`, `evicted_bytes`,
`over_budget_claims`. `/metrics`: `radiance:host_budget_bytes`, `radiance:host_used_bytes{kind}`,
`radiance:host_evictions_total{kind="prepared"|"active_copy"}`, `radiance:host_evicted_bytes_total`.

## Aktive Models und große Models

Der Store hält schon während das Model aktiv ist eine volle Host-Kopie der Gewichte (Capture beim
Load). Für große Models ist das ein zweites Mal ihre Größe im RAM; deshalb zählt diese Kopie ins
Budget und wird bei Druck zuerst verworfen (siehe oben). Das Model läuft dann weiter und ist nach der
Deaktivierung kalt.

## Grenzen

- `cache.staging_budget_mib` wird weiter nur geparst; das Staging wird gezählt, nicht begrenzt.
- Die Buchungen sind Schätzungen aus der Konfiguration (Pool-MiB, Cache-MiB, Ring, Fenster), nicht
  gemessene RSS. Gebucht wird nach dem Planen, also vor der Allokation; das Checkpoint-Store-Maß erst,
  wenn es festliegt (`--checkpoint-slots auto`).
- Mehrere Engines, die gleichzeitig aktivieren, buchen nacheinander unter dem Store-Lock; die
  Aktivierungen selbst sind wie bisher serialisiert (`build_m`).
- `admit_host_pools` rechnet mit dem, was `plan` bucht, nicht mit gemessener RSS; MemAvailable sieht, was
  andere Models schon angefasst haben, nicht was sie noch anfassen werden (gepinnter Speicher ist sofort
  belegt, der Rest eines laufenden Loads nicht).
