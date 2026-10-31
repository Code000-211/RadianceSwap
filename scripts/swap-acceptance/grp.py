#!/usr/bin/env python3
"""grp.py MODEL SECONDS : C8 diverse load on one model, return after SECONDS; then print m1's expert stats"""
import json, sys, urllib.request, time, threading, random
model, secs = sys.argv[1], float(sys.argv[2])
TOP = ["Explain how a transformer language model works.", "Write a python function that merges two sorted lists.", "Summarize the causes of the French Revolution.",
 "Translate into German: The meeting was moved to Thursday afternoon.", "What is the derivative of x^3 sin(x)?", "Give a recipe for a vegetarian lasagna.",
 "Describe the plot of Romeo and Juliet.", "Explain the difference between TCP and UDP.", "Write a haiku about the sea.", "List ten prime numbers and explain what makes them prime."]
end = time.time() + secs; done = [0]; bad = []
def post(p):
    r = urllib.request.Request("http://localhost:18080/v1/completions", json.dumps({"model": model, "prompt": p, "max_tokens": 128, "temperature": 0}).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))
def w(i):
    while time.time() < end:
        try:
            j = post(random.choice(TOP) + f" ({i})"); done[0] += j["usage"]["completion_tokens"]
        except Exception as e: bad.append(str(e))
th = [threading.Thread(target=w, args=(i,)) for i in range(8)]; [t.start() for t in th]; [t.join() for t in th]
print(model, "tokens", done[0], "errors", len(bad), bad[:2], "tok/s %.1f" % (done[0] / secs))
e = json.load(urllib.request.urlopen("http://localhost:18080/stats"))["experts"]
print("m1 experts:", {k: e[k] for k in ("flex_capacity", "movable", "promotions", "demotions", "starved_slab", "starved_pool", "no_candidate", "bytes")}, "resident", sum(e["layer_resident"]))
