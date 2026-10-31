# Slot-Abgabe eines laufenden Models (#15)

Ein laufendes Model gibt an Step-Grenzen Expert-Slots ab und nimmt sie zurück. Ablauf:
`Engine::group_tick` (nach jedem Step und im Leerlauf, höchstens alle 100 ms) meldet die Heat
(`group_heat_bins`, jede zehnte Runde), liest `GroupPlanner::target` und setzt pro Slab-Klasse
`Mover::set_resident_cap`. Der Mover führt die Kappe in `balance()` aus (ein geparkter Rank in
`idle_tick`), mit den vorhandenen Mitteln und ohne eigenen Placement-Algorithmus:

- Das budgetierte Segment des Slab schrumpft per `SlotPool::set_target` um die Differenz; die
  obersten Slots (Arbeitsreserve, kein Unit) werden nicht mehr vergeben.
- Der Host-Pool wächst um dieselbe Zahl in seine Reserve (`MoverConfig::shrinkable` reserviert
  beim Init bis zu `n_movable` Slots aus dem, was `--host-pool-mib` nach den gepoolten Units und den
  File-Tier-Reserven übrig lässt). Ein verdrängter Expert braucht dort seine Kopie.
- Die kältesten beweglichen Residents werden mit dem normalen Demote-Move verdrängt (D2H, Publish,
  Quarantäne). Ein Unit, das über dem geschrumpften Segment sitzt, wandert per Migrate nach unten.
  Danach liest die GEMM den Expert wie jeden gepoolten aus dem Host-Pool (Zero-Copy).
- Wächst die Kappe wieder, wächst das Segment, und `fill()` promotet die heißesten gepoolten Units
  in die zurückgekehrten Slots, nie in die Arbeitsreserve.
- Veröffentlicht wird nur über die normalen Moves (Retire/Quarantäne); die Residency-Tabelle wird
  wie bisher am Anfang des nächsten Steps synchronisiert. Laufende Requests bleiben unberührt.

## Wird der VRAM physisch freigegeben?

Seit #35 ja, im Manifest-Betrieb mit mehreren Models: siehe `elastic-slab.md`. Ohne Manifest gilt weiter:
Nein, nur logisch. Der Slab ist eine Reservierung im Weights-Pool, und der ist ein gewöhnlicher
`rad_dev_alloc` mit Bump-Allokator (`Pool::init`, `mem/pools.h`). Nur der KV-Pool ist elastisch
(`init_elastic`, VMem). Ein abgegebener Slot bleibt also im Pool des abgebenden Models und ist für
den Pool eines anderen Models nicht nutzbar; er ist nur ungenutzt. Eine Kappe über dem, was der
Plan platziert hat, wächst nie über die vorhandenen Slots hinaus: ein Model, das mit weniger als
seinem Anteil gestartet ist (`GroupPlanner::join`), kann nicht in den Platz hineinwachsen, den ein
anderes abgibt.

Was fehlt, damit abgegebene Bytes wirklich genutzt werden:

1. Der Expert-Slab muss auf elastischem VMem liegen (`Pools::weights` als `init_elastic`), mit
   Segmenten, die der Mover von oben nach unten entkommittet (`Pool::decommit`). Die Slot-Auswahl
   packt Residents schon nach unten (niedrigster freier Slot zuerst), sodass die obersten
   Granulate nach einem Shrink leer sind; die Kappe räumt zusätzlich Units über dem Segment ab.
2. Das aufnehmende Model muss beim Aktivieren bzw. beim Regrow gegen den freien Kartenplatz
   committen können (`Pool::commit`), nicht gegen einen beim Start fest zugeschnittenen Pool
   (`cfg_.vram_weights_mib -= cut_mib` in `Engine::plan`).
3. Eine modellübergreifende Leihe (Ticket #17, Analyse und Planner-Schicht in `docs/swap/kv-lend.md`), alternativ: der Slab des einen Models leiht
   dem des anderen Adressen (`Mover::lend_from`), wie heute der KV-Cache.

Solange diese Punkte fehlen, ist die Abgabe eine Umverteilung der Residency innerhalb der
festen Anteile (Heat statt Startreihenfolge), keine Freigabe von Bytes.
