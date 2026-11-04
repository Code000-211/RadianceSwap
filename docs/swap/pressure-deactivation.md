# Deaktivierung unter VRAM-Druck (#16)

Reichen die Slot-Abgaben (`slot-release.md`) nicht, weil die Karten für ein weiteres Model keinen
Platz mehr haben, deaktiviert der `ModelRouter` (`core/server/model_router.h`) zuerst das am längsten
idle aktive Model und aktiviert das gesuchte danach erneut. Slot-Abgaben geben keine Bytes frei
(siehe dort); nur die Deaktivierung zerstört die Engine und gibt den VRAM physisch zurück.

## Auslöser

Ein Launcher, der vor dem Attach mit `RAD_E_NOMEM` zurückkehrt (eine geworfene Exception zählt
nicht), meldet "kein Platz im VRAM". Zwei Quellen:

- vorab: `build()` in `core/main.cpp` sieht, dass `free (engste Karte) − Headroom < Anteil` ist
  (`manifest_has_room`, ein Sechzehntel des Anteils darf fehlen) und ein anderes Model VRAM hält
  (aktiv, aktivierend oder deaktivierend, `manifest_others_hold_vram`). Hält keines VRAM, gäbe es
  nichts zu deaktivieren: das Model startet mit dem, was frei ist, und sein eigenes Budget
  entscheidet. Der Anteil wurde einmal beim ersten Start gemessen; was ein fremder Prozess seitdem
  belegt, würde das Model sonst dauerhaft aussperren (#62);
- nachträglich: das Budget der Engine passt nicht (`vram_budget_resolve`) oder eine Allokation scheitert.

Jeder andere Fehler läuft wie bisher (Rollback, Backoff, bei `RAD_E_DEVICE` unhealthy). Auch der
VRAM-Fehler setzt den Backoff, aber eine wartende Anfrage umgeht ihn, wenn sie Platz schaffen kann:
sie deaktiviert und versucht es erneut. Eine spätere Anfrage, die nichts verdrängen kann, bekommt
innerhalb des Backoffs den üblichen 503 `model_activation_backoff`.

## Auswahl

Kandidat ist ein aktives Model (nicht das aktivierende), das gerade nicht benutzt wird: kein Request
zwischen Routing und Ende des Dispatch (`users`), kein Wartender, nichts im Server zugelassen
(`in_flight()`, ein Stream hält es bis zum letzten Chunk). Es wird das mit dem ältesten `busy_at`
gewählt (zuletzt in Benutzung gesehen). Ein Model mit laufenden Requests wird nie deaktiviert.

## Ablauf je wartender Anfrage

1. Ein anderes Model wird gerade deaktiviert: warten.
2. Seit dem Fehlschlag wurde ein Model fertig deaktiviert: Aktivierung neu starten.
3. Ein Kandidat existiert: deaktivieren (`pressure_deactivations`), warten.
4. Kein anderes Model aktiv, aktivierend oder deaktivierend: sofort 503 `model_activation_failed`
   ("not enough VRAM ... no other model is active").
5. Sonst sind alle anderen in Benutzung: weiter warten, bis ein Model idle wird oder die Deadline
   (`wait_ms`) mit 503 `model_activation_timeout` endet.

Höchstens so viele erneute Aktivierungen pro Anfrage, wie Models konfiguriert sind: keine Endlosschleife.

Der Launcher kehrt nach einer Deaktivierung (und nach einer zurückgerollten Aktivierung) erst
zurück, wenn der freie VRAM der Karte sich gesetzt hat (`manifest_settled_free_mib`): amdgpu gibt
den VRAM eines freigegebenen Buffers einige zehn Millisekunden nach dem Free zurück (es löscht den
Buffer vorher). Direkt nach `~Engine` gelesen, zählt die Zahl noch fast alles, was das Model hielt;
auf einer RX 9060 XT standen 8 GiB frei, wo 14 GiB waren, und die erneute Aktivierung in Schritt 2
scheiterte daran (#62). Gelesen wird alle 20 ms, bis die Zahl 300 ms lang um nicht mehr als 16 MiB
gestiegen ist, höchstens 5 s.

## VRAM-Anteile

Jede Engine hat einen festen Anteil (`manifest_headroom_mib`). Der Anteil ist

    (frei der engsten Karte beim ersten Start − gpu_headroom_mib) / min(n_models, cache.max_active_models)

`cache.max_active_models` 0 (Standard) heißt alle Models: Anteil = Gleichteilung wie vor #16, alle
passen gleichzeitig, der letzte nimmt den Rest, es gibt keine Vorab-Prüfung. Ist die Zahl kleiner
als die Zahl der Models, sind die Anteile größer, die Summe übersteigt den VRAM, und die
Aktivierung über das Limit hinaus löst die Verdrängung aus. Ein explizites `--gpu-headroom-mib`
(Kommandozeile oder `engine`) umgeht die Anteilsrechnung; dann greift nur der Fehlerpfad.

## Grenzen

- `busy_at` wird bei Dispatch-Ende, beim Attach, vom Idle-Monitor und bei jeder Auswahl gestempelt;
  ohne `idle_timeout_ms` ist die Reihenfolge nach einem langen Stream ungenau, nie unsicher.
- Zwei Models, die einander abwechselnd verdrängen, laden bei jedem Wechsel neu (warm, wenn
  vorbereitet, #12).
