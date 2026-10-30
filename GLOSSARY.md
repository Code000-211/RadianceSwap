# Radiance Model Swap

Mehrere Basismodelle hinter einem Radiance-Endpoint, die sich eine GPU-Gruppe teilen; Aktivierung und Deaktivierung ohne Prozess-Neustart.

## Language

**Model**:
Ein Basis-Artefakt (eine `.rad`-Datei) mit eigener Residency; die Einheit, die gewechselt wird.
_Avoid_: Adapter, Alias, Variante

**GPU-Gruppe**:
Die Menge der GPUs, die gemeinsam ein Model per Tensor-Parallelismus ausführen.
_Avoid_: Device-Pool, Node

**Aktives Model**:
Ein Model mit Laufzeitzustand auf einer GPU-Gruppe, das Requests ausführt; mehrere Models können gleichzeitig aktiv sein und teilen sich den VRAM.
_Avoid_: geladenes Model, Running Model

**Aktivierung**:
Ein vorbereitetes oder kaltes Model wird auf einer GPU-Gruppe aktiv, ausgelöst durch den ersten Request; andere aktive Models laufen weiter.
_Avoid_: Switch, Swap, Load

**Deaktivierung**:
Ein aktives Model gibt seine Model-Runtime nach dem Idle-Timeout oder unter VRAM-Druck frei und wird wieder ein vorbereitetes Model; laufende Generierungen werden nie abgebrochen.
_Avoid_: Switch, Evict, Sleep, Unload

**Vorbereitetes Model**:
Ein nicht aktives Model, dessen Gewichte in GPU-tauglicher Form im Host-RAM liegen.
_Avoid_: Schlafendes Model, Host-Ready

**Kaltes Model**:
Ein konfiguriertes Model ohne RAM-Backing, etwa nach LRU-Verdrängung bei vollem RAM; beim Switch wird es von Disk bzw. aus dem Page-Cache geladen.
_Avoid_: Entladenes Model

**Model-Runtime**:
Die einem aktiven Model gehörenden GPU-seitigen Ressourcen pro Rank (Streams, Pools, Bindings, Tapes); sie laufen parallel zu den Runtimes anderer Models.
_Avoid_: Engine, Instanz

**Gruppen-Heat**:
Die über alle aktiven Models einer GPU-Gruppe vergleichbare Expert-Popularität, nach der knappe Expert-Slots vergeben werden; jedes Model behält einen Mindestanteil.
_Avoid_: globale Heat, Model-Heat

**KV-Partition**:
Der Anteil eines aktiven Models am gemeinsamen KV-Byte-Budget einer GPU-Gruppe, in seinem eigenen Layout.
_Avoid_: KV-Pool, geteilter Cache

**KV-Leihe**:
Idle KV-Blöcke, die dem Expert-Slab überlassen werden und bei Bedarf durch Verdrängen von Experts zurückkehren, auch über Model-Grenzen hinweg.
_Avoid_: KV-Sharing, Loan-Pool

**RAM-Streaming**:
Ausführung eines aktiven Models, bei der Gewichte, die nicht in den VRAM passen, während der Inferenz aus dem Host-RAM nachgeladen werden.
_Avoid_: Offload, Tiering
