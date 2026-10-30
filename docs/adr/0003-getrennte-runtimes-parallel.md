# Aktive Models laufen als getrennte, parallele Runtimes

Jedes aktive Model bekommt eigene Runtime-Ressourcen (Streams, Threads, Ctx, Pools), und seine Passes laufen wirklich parallel zu denen anderer Models auf derselben GPU-Gruppe. Einen gemeinsamen Step-Loop, der zwischen den Models abwechselt, gibt es nicht. Grund: höhere GPU-Auslastung, und die Latenz eines Models hängt nicht an den Passes des anderen. Der Preis: Die All-Reduce-Kernels warten aktiv auf ein Peer-Flag (Kommentar in `core/runtime/ctx.h`), und zwei solche Ströme gleichzeitig auf TP=2 bergen das Risiko gegenseitiger Blockade sowie Konkurrenz um Link-Bandbreite.

## Consequences

- Collectives aller Models einer GPU-Gruppe laufen in einer gruppenweiten, auf allen Ranks identischen Reihenfolge. Damit gibt es keine Kreuzblockade, aber auch kein paralleles Kommunizieren.
