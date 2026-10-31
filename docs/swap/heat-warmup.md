# Heat-Warmup bei Aktivierung (#20)

Die Expert-Heat eines Models überlebt seine Deaktivierung, und die nächste Aktivierung legt die
heißesten Experts gleich beim Laden in den VRAM, innerhalb eines Byte-Budgets. Ohne gespeicherte Heat,
ohne Manifest oder mit `cache.warmup_budget_mib: 0` verhält sich alles wie zuvor.

## Was gespeichert wird

Beim Zerstören der Engine (`Engine::~Engine`, nach `stop_execution`) exportiert `HeatEngine::export_profile`
die Heat von Rank 0 (Routing ist auf allen Ranks gleich) im Format des Expert-Profils
(`RadFileProfile`: Layer, Expert, `share` = Anteil an der Gesamt-Heat; Experts ohne Heat fehlen). Das ist
ein reines Metadatum: keine Unit-Indizes, Slots oder Device-Pointer. `PreparedWeightStore::save_heat`
legt es unter dem Model-Namen ab, getrennt von den Gewichte-Einträgen: LRU-Verdrängung (kalt),
`release` und ein Schlüsselwechsel lassen es stehen, und das Host-Budget zählt es nicht. Eine Engine,
die nichts geroutet hat (dichtes Model, statischer Plan), überschreibt eine frühere Heat nicht.
Die Heat gilt nur im Prozess.

## Wie sie genutzt wird

`Engine::plan` liest sie beim ersten Rank, wenn `cache.warmup_budget_mib > 0` und ein Manifest-Registry
gesetzt ist, und gibt sie dem Planner als Profil: derselbe Pfad wie `--expert-profile` und das
Container-Profil (`PlannerInput::profile`, `rank_units`, `assign_expert_tiered`). Ein explizites
`--expert-profile` bleibt vorrangig; die gespeicherte Heat ersetzt das Container-Profil.

Das Budget (`PlannerInput::warmup_budget_bytes`, je Rank, in Slot-Bytes der Experts) schneidet das
Profil: `Planner::cut_warmup` sortiert die Experts nach `share`, behält die heißesten, solange ihre
Summe das Budget nicht übersteigt, und stoppt beim ersten, der nicht mehr passt (kein Überspringen zu
kleineren, kälteren). Alle übrigen verlieren ihren `share` und werden in der Standardreihenfolge
(expertweise) platziert. So liegen höchstens `Plan::warmup_bytes` ≤ Budget an Warmup-Experts vorn im
Ranking; wie viele davon wirklich in den VRAM passen, entscheidet wie bisher der Budget-Fit des
Planners. Der Planner meldet `warmup_units`/`warmup_bytes`, die Engine loggt sie.

Die HeatEngine startet wie bisher bei Heat 0 und lernt aus dem echten Routing; die gespeicherte Heat
bestimmt nur die Startresidenz.

## Messpunkt: Trefferrate nach der Aktivierung

`HeatStats::early_routed` / `early_routed_resident` zählen geroutete bzw. residente Experts nur in den
ersten `HeatParams::early_steps` (64) Steps nach dem Start der Engine. Sie stehen in `/stats` (Engine-
Snapshot, `experts.early_routed`, `early_routed_res`) und als `early_hit=…%` im Live-View.
Gemessen in [gpu-acceptance.md](gpu-acceptance.md): dasselbe Model mit `warmup_budget_mib: 0` und mit
Budget deaktivieren/reaktivieren und `early_hit` vergleichen (83,45 % gegen 89,12 %).

## Grenzen

- Die Heat ist die abklingende Heat des Deaktivierungsmoments (Halbwertszeit ~14 Steps), also die
  letzte Konversation des Models.
- Unter TP ist das Budget je Rank; bei ungleichen Shards kann die Auswahl je Rank leicht abweichen.
- Das Budget begrenzt die vorn gerankten Experts, nicht den VRAM insgesamt: der Rest des Pools wird
  wie ohne Warmup gefüllt.
