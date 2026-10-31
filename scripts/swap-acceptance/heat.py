#!/usr/bin/env python3
"""heat.py PHASE : run the topic workload on model moe, print early_hit stats"""
import json, sys, urllib.request, time
TOPICS = ["Explain how a transformer language model works.", "Write a python function that merges two sorted lists.",
 "Summarize the causes of the French Revolution.", "Translate into German: The meeting was moved to Thursday afternoon.",
 "What is the derivative of x^3 sin(x)?", "Give a recipe for a vegetarian lasagna.",
 "Describe the plot of Romeo and Juliet in five sentences.", "Explain the difference between TCP and UDP."]
def post(p, n=96):
    r = urllib.request.Request("http://localhost:18080/v1/completions", json.dumps({"model": "moe", "prompt": p, "max_tokens": n, "temperature": 0}).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))
def stats(): return json.load(urllib.request.urlopen("http://localhost:18080/stats", timeout=30))
t = time.time()
for p in TOPICS: post(p)
print("workload %.1fs" % (time.time() - t))
s = stats(); e = s.get("experts", {})
print({k: v for k, v in e.items() if "early" in k or k in ("hit_rate", "resident_hit")})
er, ers = e.get("early_routed"), e.get("early_routed_res")
if er: print("early_hit = %.2f%%  (%s of %s)" % (100.0 * ers / er, ers, er))
print("hit-ish keys:", [k for k in e if "hit" in k or "res" in k][:12])
