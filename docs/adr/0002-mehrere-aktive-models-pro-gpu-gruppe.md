# Mehrere aktive Models teilen sich eine GPU-Gruppe

Abweichend von Issue #10, das ein aktives Model pro GPU-Gruppe vorsieht, dürfen mehrere Models gleichzeitig aktiv sein. Sie teilen sich den VRAM, und was nicht hineinpasst, läuft per RAM-Streaming. Grund: Bei gemischter Last soll ein Request für B nicht auf die längste Generierung von A warten müssen, und laufende Generierungen werden nie abgebrochen. Der Preis: mehrere gleichzeitige Runtimes, eine Aufteilung von VRAM und KV-Cache und Konkurrenz um Streams und Collectives.

## Consequences

- RAM-Streaming muss auf statische Gewichte erweitert werden, also über die heutigen Experts hinaus (Zero-Copy und Stager-Ring), damit ein neues Model auch bei vollem VRAM sofort läuft.
- Ein laufendes Model gibt an Step-Grenzen Expert-Slots ab (SlotPool-Shrink), wenn ein anderes Model aktiviert wird.
- Die bestehende KV-Leihe (idle KV-Blöcke gehen an den Expert-Slab, Rückforderung verdrängt Experts) wird modellübergreifend: Die KV-Partition eines Models kann Platz aus dem Expert-Slab eines anderen Models zurückfordern. Laufende Sequenzen werden nie verdrängt.
- Knappe Expert-Slots vergibt ein gruppenweiter Planner nach Gruppen-Heat (gewichtet nach Bytes), mit einem Mindestanteil pro Model.
- Aktivierung beim ersten Request, Deaktivierung nach Idle-Timeout oder unter VRAM-Druck (zuerst das am längsten idle Model). Die Zahl aktiver Models ist nur durch das Budget begrenzt.

## Considered Options

- Ein aktives Model mit Drain vor dem Switch (Issue #10): verworfen, weil die Wartezeit bei gemischter Last unbegrenzt sein kann.
- Überlappung nur während des Switch: verworfen zugunsten der allgemeinen Lösung.
