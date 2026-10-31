# Warme Aktivierung aus dem RAM (#12)

Bei der Deaktivierung bleiben die Gewichte eines Models im Host-RAM; das Model ist dann ein
**vorbereitetes Model** (Zustand `prepared`, Status `/radiance/models`), nicht kalt. Die nächste
Aktivierung lädt sie per begrenztem gepinntem Staging-Ring hoch und liest weder Checkpoint noch
Container, parst nichts, relayoutet nichts und kopiert nichts von der GPU zurück. Gewichte werden
nie dequantisiert oder requantisiert: gespeichert wird genau die Form, die die Kernel lesen.

## Woher kommt das Backing?

`core/model/weight_store.{h,cpp}` (`PreparedWeightStore`, ein Objekt pro Prozess, in
`serve_manifest` vor Engines und Router angelegt). Die **erste** Aktivierung füllt ihn, ohne
zusätzlichen Disk-Read und ohne D2H:

- **Checkpoint** (HF/GGUF direkt): der Relayout-Pool (`RelayoutPool::run`) hat die Endform ohnehin in
  einem Host-Puffer (Pinned-Slot, `tmp` oder der Host-Pool-Slot) und kopiert sie zusätzlich in den
  Store-Puffer (`StagedWeight::cap`).
- **Container (`.rad`)**: das Sweep liest wie bisher (O_DIRECT) und kopiert in den Slot. Für ein
  Gewicht, das der Store noch nicht hält, legt `load_weights` einen zweiten Run mit Host-Ziel
  (Store-Puffer) an; die Pool-Worker kopieren daraus parallel aus demselben Lese-Fenster. Die
  Lesemenge bleibt gleich.

Die warme Aktivierung läuft über dieselbe Planung (Plan, Slots, Zielpointer) und ersetzt nur die
Quelle: ein `StagedWeight` mit `warm` wird vom Pool-Worker über `Bounce::upload` (Ring aus 8
gepinnten 4-MiB-Slots je Worker, bei rein warmer Aktivierung höchstens 4 Worker, also höchstens
128 MiB gepinnt) in den VRAM- oder Host-Pool-Slot geschrieben.

### RAM-Backing vs. Page-Cache-Backing

Jeder Store-Eintrag ist **RAM-Backing**: eine anonyme Allokation des Prozesses. Nur sie wird später
vom Host-Budget (#13) gezählt und verdrängt. Nicht im Store (und damit nicht „vorbereitet"):

- das **Mapped-Tier** (z. B. n-gram-Tabelle, die in place aus dem Container-Mapping gelesen wird) und
  das **File-Tier**. Sie gehören der Engine, die den Container mappt, und kommen bei jeder Aktivierung
  aus dem Page-Cache bzw. von Disk (`mapped_bytes` im Report). Sie sind Page-Cache-Backing: der Kernel
  kann sie jederzeit verdrängen.
- unveränderte Container-Bytes werden **nicht** als mmap-Referenz geführt, sondern kopiert. Eine
  Referenz würde RAM sparen, wäre aber Page-Cache-Backing und könnte die Zusage „zweite Aktivierung
  liest keine Checkpoint-Bytes" nicht halten.

Der Store legt die Kopie nur an, wenn das ganze Model neben `--host-headroom-mib` und den cgroup-Limits
in den Host-RAM passt (`lockable_room`); sonst bleibt das Model kalt (Warnung im Log). Das ist eine
Schutzgrenze; Budget und LRU-Verdrängung: [host-budget.md](host-budget.md) (#13).

## Schlüssel

Modell-Schlüssel (`weight_store_key`): Artefakt-Identität (kanonischer Pfad plus Größe und mtime je
Datei, bei einem Verzeichnis jeder Datei darin), Build (`RAD_VERSION`, Kernel-Hierarchie), `--tp`,
`--placement`, `--weights-disk-tier`. Ein Load unter anderem Schlüssel verwirft die Einträge
des Models (kalt). Eintrag je Gewicht (`weight_store_entry`): Rank, Name, Quelle, Encoding,
Layout-Tag, identity/widen/narrow, gespeicherter dtype, Größe und Shard. Ein Gewicht, dessen Form
anders ist, wird kalt geladen und ersetzt den Eintrag. Kommt die Aktivierung mit anderem Plan (anderes
Tier), nutzt sie die Einträge, die passen, und liest den Rest von der Quelle (dann „kalt").

Der Store gilt nur im Prozess; nach einem Neustart sind alle Models kalt.

## Status und Metriken

Pro Aktivierung meldet die Engine einen `WeightLoadReport` (`Engine::weight_load_report`), den der
Launcher an den Router gibt (`note_weight_load`):

- `/radiance/models`: `state` (`prepared` nach der Deaktivierung), `prepared_bytes`,
  `warm_activations`, `cold_activations`, `last_weight_load` mit `kind` (`warm`/`cold`), `bytes`,
  `bytes_from_store`, `seconds`, `source_read_bytes` (Zähler des Loaders: Checkpoint- bzw.
  Container-Bytes), `disk_read_bytes` (`read_bytes`-Delta aus `/proc/self/io` während des Loads,
  inklusive Page-Cache-Misses und mmap-Faults, **prozessweit**: parallele Aktivierungen anderer
  Models erscheinen darin; -1, wenn nicht lesbar) und `mapped_bytes`.
- `/metrics`: `radiance:model_prepared_bytes`, `radiance:model_weight_loads_total`,
  `…_load_bytes_total`, `…_load_seconds_total`, `…_source_read_bytes_total`,
  `…_disk_read_bytes_total`, je mit `kind="warm"|"cold"`.

Warm heißt: jedes in einen Pool gelegte Gewicht kam aus dem Store. Header-Metadaten des Checkpoints
(Tensor-Index, config.json) liest jeder Start weiterhin, um zu deklarieren und zu planen; das sind
wenige KiB und tauchen nur in `disk_read_bytes` auf, nicht in `source_read_bytes`.

## Grenzen

- Nur `Engine::load_weights` (die deklarierten Gewichte). Encoder-Türme, Drafter und n-gram-Tabellen
  im Mapped-Tier laden wie bisher.
- Ohne `cache.host_budget_mib` hält jedes deaktivierte Model seine Gewichte, solange die Host-Checks
  es zulassen (dann verdrängt der Store das am längsten ungenutzte, `host-budget.md`).
- Der Host-Test deckt die Upload-Strecke zur GPU (`Bounce::upload`) nicht ab; sie ist dieselbe wie
  im bisherigen Identity-Pfad.
