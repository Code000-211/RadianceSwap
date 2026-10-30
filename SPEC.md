## Problem Statement

Ein Radiance-Prozess bedient heute genau ein Model. Wer mehrere große Models (Dense oder MoE) auf derselben GPU-Gruppe anbieten will, muss Prozesse von Hand stoppen und starten. Bei jedem Neustart laufen Checkpoint-Parsing, Relayout, Tokenizer- und Tape-Aufbau erneut, und der Endpoint ist während dieser Zeit weg. Bei gemischter Last, wenn mehrere Clients gleichzeitig verschiedene Models anfragen, ist das unbrauchbar: Ein Request für Model B wartet, bis Model A beendet wurde. Laufende Generierungen von A abzubrechen ist nicht akzeptabel.

## Solution

Ein Radiance-Prozess bedient mehrere konfigurierte Models über einen Endpoint. Mehrere Models können gleichzeitig **aktiv** sein und teilen sich den VRAM der GPU-Gruppe. Jedes aktive Model läuft als eigene **Model-Runtime**, parallel zu den anderen.

- Ein Request für ein inaktives Model löst dessen **Aktivierung** aus. B läuft sofort. Was nicht in den VRAM passt, kommt per **RAM-Streaming** aus dem Host-RAM, und zwar Experts *und* statische Gewichte.
- Laufende Models geben an Step-Grenzen Expert-Slots ab. Knappe Slots vergibt die **Gruppen-Heat**, jedes Model behält einen Mindestanteil.
- Die bestehende **KV-Leihe** funktioniert modellübergreifend. Braucht die **KV-Partition** eines Models Platz, werden Experts verdrängt, egal welchem Model sie gehören. Laufende Sequenzen werden nie verdrängt.
- Ein Model ohne Requests wird nach einem Idle-Timeout oder unter VRAM-Druck **deaktiviert** und bleibt als **vorbereitetes Model** im RAM. Ist der RAM voll, wird das am längsten ungenutzte Model zum **kalten Model**.
- Laufende Generierungen werden nie abgebrochen.
- Ohne Manifest verhält sich Radiance exakt wie heute.

Begriffe: siehe `GLOSSARY.md`. Entscheidungen: ADR 0001 (In-Process), 0002 (mehrere aktive Models), 0003 (parallele Runtimes, gruppenweite Reihenfolge der Collectives).

## User Stories

1. Als Betreiber möchte ich mehrere Models in einem Manifest konfigurieren, damit ein Radiance-Prozess sie alle bedient.
2. Als Betreiber möchte ich, dass `--model` ohne Manifest unverändert funktioniert, damit bestehende Deployments nicht brechen.
3. Als Client möchte ich das Model über das Feld `model` des OpenAI-Requests wählen, damit ich keinen zweiten Endpoint brauche.
4. Als Client möchte ich, dass `/v1/models` alle konfigurierten Models auflistet, damit ich weiß, was angeboten wird.
5. Als Client möchte ich für eine unbekannte Model-ID einen eindeutigen Fehler erhalten, damit ich Tippfehler sofort erkenne.
6. Als Client möchte ich, dass ein Request ohne `model` das konfigurierte Default-Model nutzt, wo der Endpoint das erlaubt.
7. Als Client möchte ich, dass mein Request Tokenizer, Chat-Template, Grammatik und Kontextlimit *seines* Models nutzt, damit keine Verwechslung zwischen Models möglich ist.
8. Als Client von Model B möchte ich, dass mein Request sofort bearbeitet wird, auch während Model A lange Generierungen ausführt, damit gemischte Last nicht blockiert.
9. Als Client von Model A möchte ich, dass meine laufende Generierung nie wegen eines anderen Models abgebrochen wird.
10. Als Client möchte ich, dass gleichzeitige erste Requests für dasselbe inaktive Model nur eine Aktivierung auslösen.
11. Als Client möchte ich einen Request abbrechen können, während er auf eine Aktivierung wartet, ohne dass Ressourcen lecken.
12. Als Client möchte ich bei einer fehlgeschlagenen Aktivierung einen klaren Fehler erhalten statt eines hängenden Requests.
13. Als Betreiber möchte ich, dass ein Model beim ersten Request aktiviert wird, damit kein VRAM für ungenutzte Models reserviert ist.
14. Als Betreiber möchte ich einen Idle-Timeout pro Konfiguration, nach dem ein Model ohne Requests deaktiviert wird.
15. Als Betreiber möchte ich, dass unter VRAM-Druck zuerst das am längsten idle Model deaktiviert wird.
16. Als Betreiber möchte ich, dass deaktivierte Models ihre Gewichte in GPU-tauglicher Form im RAM behalten, damit die nächste Aktivierung ohne Parsing, Konvertierung oder D2H-Kopie auskommt.
17. Als Betreiber möchte ich ein explizites Host-RAM-Budget, damit vorbereitete Models plus aktive Host-Pools nie mehr belegen als erlaubt (inklusive cgroup- und Pin-Limits).
18. Als Betreiber möchte ich, dass bei vollem RAM das am längsten ungenutzte vorbereitete Model per LRU kalt wird, statt dass der Start oder die Aktivierung scheitert.
19. Als Betreiber möchte ich sehen, ob eine Aktivierung warm (aus dem RAM) oder kalt (Disk/Page-Cache) war, inklusive der tatsächlich gelesenen Disk-Bytes.
20. Als Betreiber möchte ich, dass Gewichte beim Aktivieren nie stillschweigend dequantisiert oder requantisiert werden.
21. Als Betreiber möchte ich, dass die Zahl aktiver Models nur durch das Budget begrenzt ist, nicht durch ein festes Limit.
22. Als Betreiber möchte ich, dass ein Model auch dann aktiv wird, wenn seine statischen Gewichte nicht in den freien VRAM passen. Die fehlenden Teile werden per RAM-Streaming geladen.
23. Als Betreiber möchte ich, dass ein laufendes Model bei der Aktivierung eines anderen Expert-Slots abgibt, statt dieses warten zu lassen.
24. Als Betreiber möchte ich, dass knappe Expert-Slots nach der Gruppen-Heat über alle aktiven Models vergeben werden, damit der VRAM dorthin geht, wo er am meisten bringt.
25. Als Betreiber möchte ich einen Mindestanteil pro aktivem Model, damit kein Model verhungert.
26. Als Betreiber möchte ich, dass die KV-Partition eines Models Platz durch Verdrängen von Experts *anderer* Models zurückholen kann, wie heute innerhalb eines Models.
27. Als Betreiber möchte ich, dass laufende Sequenzen bei VRAM-Umverteilung nie ihren KV-Cache verlieren.
28. Als Betreiber möchte ich, dass sich parallele Collectives mehrerer Models auf TP=2 nie gegenseitig blockieren.
29. Als Betreiber möchte ich, dass ein Model nach der Deaktivierung keinen VRAM mehr hält, abgesehen von ausgewiesenem, persistentem Device- und Plugin-Overhead.
30. Als Betreiber möchte ich, dass auch hunderte Aktivierungen und Deaktivierungen keinen monotonen VRAM- oder Host-Speicherzuwachs erzeugen.
31. Als Betreiber möchte ich, dass eine fehlgeschlagene Aktivierung (OOM, Rank-Fehler, zu wenig RAM) alle Teilallokationen auf allen Ranks freigibt und die anderen aktiven Models unberührt lässt.
32. Als Betreiber möchte ich, dass ein Device-Fehler die GPU-Gruppe als unhealthy markiert, statt Aktivierungen endlos zu wiederholen.
33. Als Betreiber möchte ich, dass die Expert-Heat eines deaktivierten Models als Warmup-Hinweis erhalten bleibt, damit die nächste Aktivierung schneller wieder eine gute Trefferrate erreicht.
34. Als Betreiber möchte ich ein begrenztes Byte-Budget für das Vorladen heißer Experts bei der Aktivierung.
35. Als Betreiber möchte ich pro Model die bestehenden, validierten Engine-Optionen (KV-dtype, Kontext, Speculation/MTP, Placement) im Manifest setzen.
36. Als Betreiber möchte ich im Status/Admin pro Model sehen: Zustand (kalt, vorbereitet, aktivierend, aktiv, deaktivierend, fehlgeschlagen), Queue-Tiefe, ältestes Wartealter, VRAM-Anteil, KV-Partition und Anteil an RAM-Streaming.
37. Als Betreiber möchte ich Metriken für Aktivierungen und Deaktivierungen (Anzahl, Fehler, Dauern der einzelnen Phasen, übertragene Bytes, pinned und ordinary Host-Bytes, freigegebene Bytes pro GPU).
38. Als Betreiber möchte ich TTFT aufgeteilt in Queue, Aktivierung und Prefill sehen, mit Model-Identität in jeder Metrik.
39. Als Betreiber möchte ich messen können, wie stark RAM-Streaming und VRAM-Teilung den Decode- und Prefill-Durchsatz jedes Models drücken.
40. Als Betreiber möchte ich eine Doku zu RAM-Sizing, warmer vs. kalter Aktivierung und den Kosten von RAM-Streaming.
41. Als Betreiber möchte ich, dass LoRA-Konfiguration zusammen mit einem Manifest beim Start explizit abgelehnt wird, statt mehrdeutig zu routen.
42. Als Betreiber möchte ich, dass Dense- und MoE-Models gleichermaßen unterstützt werden.
43. Als Betreiber möchte ich, dass bestehende Single-Model-Features, auch Multimodal, ohne Manifest weiter funktionieren.
44. Als Client möchte ich, dass die Ausgabe eines Models mit identischer Konfiguration, Placement und Seed deterministisch dieselbe ist wie im Single-Model-Betrieb.
45. Als Entwickler möchte ich, dass Prozess-Ressourcen (Plugin/Tools-Registry, HTTP-Listener) dem Prozess gehören und nicht einer Model-Runtime, damit eine Deaktivierung nichts schließt, was andere Models nutzen.
46. Als Entwickler möchte ich, dass Sinks und Frontends eines Requests bis zum Ende seines Response-Lebenszyklus gültig bleiben, auch über eine Deaktivierung hinweg.

## Implementation Decisions

- **Prozessmodell (ADR 0001):** Alles läuft in einem Prozess. HTTP-Listener, Plugin- und Tools-Registry sowie die Koordination der GPU-Gruppe gehören einer Prozess-Runtime. Model-Runtimes können sie nicht schließen.
- **Lifecycle-Refactor der Engine:** `serve()` wird in Ausführungs-Start/-Stop und unabhängiges HTTP-Serving getrennt. Die bisherigen stack-lokalen Bridges in `run_server()` werden zu einem lebensdauer-verwalteten Model-Frontend (Tokenizer, Template, Grammatik, Limits). Die bestehende Reihenfolge beim Teardown bleibt erhalten (Mover-Transfers drainen vor dem Ctx-Shutdown). Teilaktivierungen lassen sich idempotent aufräumen.
- **Neue Module:**
  - *ModelRegistry/Manifest:* Parst das Manifest, prüft eindeutige IDs und Default-Model, lehnt die Kombination mit LoRA ab und hält einen ModelRecord pro Model (Identität und Version des Artefakts, Engine-Optionen, Frontend, Zustand, Heat-Metadaten).
  - *ModelManager:* Löst die Model-ID eines Requests auf, *bevor* tokenisiert oder das Template angewendet wird. Er führt pro Model eine begrenzte Queue und einen einzigen Aktivierungs-Task, verwaltet Idle-Timeout und Deaktivierung und vergibt Leases, die das Frontend bis zum Ende der Response gültig halten.
  - *PreparedWeightStore:* Host-Backing in GPU-tauglicher gepackter Form, gekeyt nach Artefakt-Identität, Rank/Shard, Layout/Encoding und Build-Identität. Er nutzt die bestehende Pipeline aus Load und Relayout und unterscheidet echtes RAM-Backing von Page-Cache-Backing. Für Uploads gibt es einen begrenzten gepinnten Staging-Ring. Verdrängung geschieht per LRU und macht ein Model kalt.
  - *Gruppen-Planner:* Eine gruppenweite Instanz über dem bestehenden Planner/Mover. Er vergibt Expert-Slots nach Gruppen-Heat (gewichtet nach Bytes) mit einem Mindestanteil pro Model und rechnet bei jeder Aktivierung und Deaktivierung die Budgets pro Rank neu. Einen zweiten Placement-Algorithmus gibt es nicht.
- **Zustände pro Model:** `KALT → VORBEREITEND → VORBEREITET → AKTIVIEREND → AKTIV → DEAKTIVIEREND → VORBEREITET`. Fehlschläge enden nach dem Aufräumen in `FEHLGESCHLAGEN`. Ein Device-Fehler setzt die GPU-Gruppe auf unhealthy. Ein Model gilt erst als `AKTIV`, wenn alle Ranks bereit sind.
- **Parallele Model-Runtimes (ADR 0003):** Jedes aktive Model hat eigene Streams, Threads, Ctx und Pools pro Rank. Collectives aller Models einer Gruppe laufen in einer gruppenweiten, auf allen Ranks identischen Reihenfolge (Ticket/Ordnung pro Gruppe). Jede Runtime hat eigene Flag- und Collective-Puffer.
- **RAM-Streaming für statische Gewichte (neu):** Die bestehenden Wege für Experts, Zero-Copy aus dem gepinnten Host-Pool und der Stager-Ring, werden auf statische Gewichte erweitert. Ein Model kann damit mit fast leerem VRAM-Anteil aktiv werden und wandert in den VRAM, sobald Slots frei werden.
- **VRAM-Umverteilung:** Laufende Models geben Expert-Slots an Step-Grenzen ab (Shrink und Regrow des SlotPool). Weight-Tabellen und Tapes einer Runtime werden nur an diesen Grenzen neu publiziert.
- **Modellübergreifende KV-Leihe:** Ein gemeinsames KV-Byte-Budget pro Gruppe, aufgeteilt in Partitionen pro Model, jede im eigenen Layout. Idle Blöcke werden wie heute an Expert-Slabs verliehen, jetzt auch an Slabs anderer Models. Zurückgeholt wird durch Verdrängen von Experts. Laufende Sequenzen werden nie verdrängt.
- **Idle-Cache-Policy v1:** Bei der Deaktivierung werden Prefix-, KV- und Recurrent-State verworfen. Tokenizer, Template, Konfiguration, Gewichte und Heat-Metadaten bleiben erhalten.
- **API:** `/v1/models` listet alle Models. Residency und Zustand stehen in Admin/Status. Unbekannte IDs werden im Multi-Model-Modus mit einem Fehler abgelehnt.
- **Konfiguration (neu, Vorschlag):** Ein Manifest mit einem Abschnitt für die GPU-Gruppe (Devices, TP), einem Cache-Abschnitt (Host-Budget, Staging-Budget, Idle-Timeout, Queue-Limits, Warmup-Budget) und einer Model-Liste (ID, Pfad, Default-Flag, Engine-Optionen pro Model).

## Testing Decisions

- Gute Tests prüfen äußeres Verhalten: Was ein Client oder Betreiber sieht (Antworten, Fehler, Zustände, Metriken, Slot- und Byte-Zuteilung), nicht interne Datenstrukturen.
- **Seam 1, HTTP:** Tests laufen über `Router::dispatch` mit den bestehenden Server-Fakes. Dahinter steht der echte ModelManager mit Fake-Model-Runtimes (eine Fabrik, die ein Fake-Scheduler/-Frontend pro Model liefert). Abgedeckt:
  - Routing zum richtigen Frontend, unbekannte IDs, Default-Model
  - genau eine Aktivierung bei gleichzeitigen Requests
  - Abbruch während des Wartens
  - Idle-Timeout, LRU zum kalten Model, Fehlerpfade und Rollback
  - Lifetimes von Sinks und Leases über eine Deaktivierung hinweg
  - Metriken und Status
  - LoRA zusammen mit Manifest wird abgelehnt
  - Single-Model-Pfad unverändert

  Vorbild: `server_test`, `lifecycle_test` (Reap-Vertrag), `server_fakes.h`, `sched_fakes.h`.
- **Seam 2, Planner:** Der Gruppen-Planner wird direkt getestet:
  - Vergabe nach Gruppen-Heat, Mindestanteil
  - Slot-Abgabe eines laufenden Models
  - modellübergreifende Rückforderung durch die KV-Leihe
  - Budget-Neuberechnung bei Aktivierung und Deaktivierung
  - Fallback auf RAM-Streaming für statische Gewichte, wenn der VRAM nicht reicht

  Vorbild: `place_test` (z. B. Fit-, Reporting- und Sharding-Fälle unter World Size 2), `hostmem_test` für das Host-Budget.
- **Seam 3, GPU-Abnahme:** `engine_e2e_test` und dedizierte Läufe auf 2× R9700 mit TP=2:
  - Kombinationen Dense/Dense, Dense/MoE und MoE/MoE, gleichzeitig aktiv
  - mindestens 100 Zyklen Aktivierung und Deaktivierung ohne monotones Leck
  - deterministische Parität zu Single-Model mit identischem Placement und Seed
  - Greedy und Sampling, MTP, Streaming, Abbruch
  - Collectives unter paralleler Last ohne Hänger
  - Messung der Kosten von RAM-Streaming für statische Gewichte

## Out of Scope

- LoRA-Adapter, Control-Vectors und Adapter-Aliase im Multi-Model-Modus. LoRA bleibt im Single-Model-Betrieb unverändert.
- Abbruch oder Preemption laufender Generierungen.
- `idle_cache_policy: preserve_host`, also KV- und Recurrent-Snapshots über eine Deaktivierung hinweg. Das ist eine spätere Erweiterung.
- Mehrere GPU-Gruppen mit Migration von Models zwischen ihnen.
- Multimodal- und Audio/Video-Endpoints im Multi-Model-Modus. Sie werden dort klar abgelehnt.
- Harte Latenzgarantien und Ziele für die Aktivierungszeit, solange keine Messungen vorliegen.

## Further Notes

- Ursprung: Code000-211/Radiance#10. Diese Spec weicht bewusst ab: Statt eines aktiven Models mit Drain beim Switch können mehrere Models gleichzeitig aktiv sein (ADR 0002).
- Größtes Risiko: Der Durchsatz beim RAM-Streaming statischer Gewichte über PCIe ist ungemessen. Er sollte früh in einem Prototyp gemessen werden.
- Zweitgrößtes Risiko: Die gruppenweite Reihenfolge der Collectives serialisiert die Kommunikation. Unter paralleler Last ist zu messen, wie viel Parallelität übrig bleibt.
- Die Code-Bezüge stammen von `main@ceff2ae`.
