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
  Pools aktiver Models werden nie verdrängt: bleibt es trotzdem über dem Budget, zählt
  `over_budget_claims` und der Log warnt. Die Aktivierung scheitert deshalb nicht am Budget, sondern
  nur an den bestehenden Host-Checks.
- Ein Model, dessen Load läuft (`begin` … `commit`/`abort`), wird nie verdrängt (der Load hält Zeiger
  in seine Einträge).
- **Ohne Budget** (0) entscheiden die bestehenden Checks (`lockable_room`: MemAvailable,
  cgroup-Ebenen v1/v2, bei gelockten Kopien RLIMIT_MEMLOCK, `--host-headroom-mib`). Sie laufen über
  `Engine::host_room`: lehnt `lockable_room` ab und hält der Store Kopien, wird die am längsten
  ungenutzte verdrängt und neu geprüft. Auch mit Budget gelten diese Checks zusätzlich. Pinned
  Speicher, den die Karte liest, zählt (wie bisher) nicht gegen RLIMIT_MEMLOCK.
- Das Verdrängen gibt den Speicher sofort zurück: Store-Puffer sind anonyme `mmap`s (`munmap`), kein
  malloc-Arena, die ihn behalten könnte.

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
