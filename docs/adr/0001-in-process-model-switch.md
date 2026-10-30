# Aktivierung und Deaktivierung von Models innerhalb eines Radiance-Prozesses

Aktivierung und Deaktivierung von Models passieren innerhalb eines einzigen Radiance-Prozesses. Einen externen Router, der pro Model einen Prozess neu startet, verwenden wir nicht. Zwei Gründe: Tokenizer, Relayout und Tape-Aufbau sollen nicht bei jeder Aktivierung neu laufen (Latenz), und es soll einen Endpoint ohne zusätzliche Proxy-Komponente geben (Betrieb). Der Preis ist ein Lifecycle-Refactor der Engine (`serve()` aufteilen, Ownership der Registry, Frontend-Lifetimes).

## Considered Options

- Externer Router mit Prozess-Neustart pro Model, Gewichte warm im Page-Cache: verworfen wegen der Switch-Latenz und der zusätzlichen Komponente.
