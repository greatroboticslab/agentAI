#!/usr/bin/env bash
# End-to-end test of the STUDENT dataset upload + analysis pipeline.
# Runs on the lab (needs ~/.dash_session_key + the .venv). Walks the FULL journey
# as a real member + every upload path + edge cases. Exits non-zero on any failure.
set -u
BASE="${BASE:-http://127.0.0.1:8000}"
PY="$HOME/weed_llm_benchmark/.venv/bin/python"
U="${DASH_USER:-1}"; P="${DASH_PASS:-1}"; BA="-u $U:$P"
pass=0; fail=0
ck(){ if [ "$3" = "$2" ]; then pass=$((pass+1)); printf "  ok    %-46s %s\n" "$1" "$3";
      else fail=$((fail+1)); printf "  FAIL  %-46s got=%s exp=%s\n" "$1" "$3" "$2"; fi; }
ckin(){ case ",$2," in *,"$3",*) pass=$((pass+1)); printf "  ok    %-46s %s\n" "$1" "$3";;
        *) fail=$((fail+1)); printf "  FAIL  %-46s got=%s exp_one_of=%s\n" "$1" "$3" "$2";; esac; }
HC(){ curl -s -o /dev/null -w '%{http_code}' --max-time 60 "$@"; }
jget(){ $PY -c "import json,sys;d=json.load(sys.stdin);print(eval(sys.argv[1]))" "$1" 2>/dev/null; }
CK="Cookie: agentai_session=$($PY - <<PY
import base64,hmac,hashlib,json,time,os
k=open(os.path.expanduser("~/.dash_session_key"),"rb").read();b=lambda x:base64.urlsafe_b64encode(x).decode().rstrip("=")
p={"uid":"e2e@uni.edu","email":"e2e@uni.edu","name":"E2E","exp":time.time()+3600}
bd=b(json.dumps(p,separators=(",",":")).encode());print(bd+"."+b(hmac.new(k,bd.encode(),hashlib.sha256).digest()))
PY
)"
mem(){ curl -s -H "$CK" "$@"; }
echo "E2E dataset pipeline -> $BASE"

# ---- build payloads ----
$PY - <<'PY'
from PIL import Image
import os,random,zipfile,tarfile,csv
random.seed(7); R="/tmp/e2e"; os.system("rm -rf "+R); os.makedirs(R,exist_ok=True)
def mkcls(base, per):
    for split in ("train","val"):
        for cls,(n) in per.items():
            d=f"{base}/images/{split}/{cls}"; os.makedirs(d,exist_ok=True)
            for i in range(n if split=="train" else max(1,n//3)):
                Image.new("RGB",(48,48),(random.randint(0,255),)*3).save(f"{d}/{cls}_{split}_{i}.png")
# balanced classification (wrapper 'images/')
mkcls(f"{R}/bal", {"catA":10,"catB":10})
with zipfile.ZipFile(f"{R}/bal.zip","w") as z:
    for dp,_,fs in os.walk(f"{R}/bal"):
        for f in fs: z.write(os.path.join(dp,f), os.path.relpath(os.path.join(dp,f), f"{R}/bal"))
with tarfile.open(f"{R}/bal.tgz","w:gz") as t: t.add(f"{R}/bal", arcname=".")
# imbalanced classification
mkcls(f"{R}/imb", {"big":30,"small":3})
with zipfile.ZipFile(f"{R}/imb.zip","w") as z:
    for dp,_,fs in os.walk(f"{R}/imb"):
        for f in fs: z.write(os.path.join(dp,f), os.path.relpath(os.path.join(dp,f), f"{R}/imb"))
# unlabeled (flat images, no classes)
os.makedirs(f"{R}/unl/images",exist_ok=True)
for i in range(8): Image.new("RGB",(50,50),(9,9,9)).save(f"{R}/unl/images/u{i}.png")
with zipfile.ZipFile(f"{R}/unl.zip","w") as z:
    for dp,_,fs in os.walk(f"{R}/unl"):
        for f in fs: z.write(os.path.join(dp,f), os.path.relpath(os.path.join(dp,f), f"{R}/unl"))
# single image + loose files
Image.new("RGB",(64,64),(10,180,10)).save(f"{R}/single.png")
for n in ("l1.png","l2.png","l3.png"): Image.new("RGB",(40,40),(5,5,5)).save(f"{R}/{n}")
# non-image: a sensor csv + text
os.makedirs(f"{R}/sen",exist_ok=True)
with open(f"{R}/sen/imu.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["t","ax","ay"]); [w.writerow([i,i*0.1,i*0.2]) for i in range(200)]
open(f"{R}/sen/notes.md","w").write("sensor run notes "*50)
with tarfile.open(f"{R}/sen.tgz","w:gz") as t: t.add(f"{R}/sen", arcname="sen")
open(f"{R}/bad.txt","w").write("this is not an archive or image")
print("payloads built")
PY

echo "== 0. new student identity =="
ck "member (not admin, no cluster)" "False|False" "$(mem "$BASE/api/me" | $PY -c "import json,sys;d=json.load(sys.stdin);print('%s|%s'%(d.get('is_admin'),d.get('can_use_cluster')))")"

echo "== 1. create project =="
mem -X POST -H 'Content-Type: application/json' -d '{"domain":"e2e_ds"}' "$BASE/api/agent/delete" >/dev/null
ck "create project" 200 "$(HC -H "$CK" -X POST -H 'Content-Type: application/json' -d '{"name":"e2e ds","research_field":"ag","modality":["image"]}' "$BASE/api/agent/create")"

up_raw(){ mem --max-time 90 -X POST -H 'Content-Type: application/octet-stream' --data-binary @"$1" "$BASE/api/dataset/upload?domain=e2e_ds&name=$2${3:-}" | jget "d.get('slug','')"; }
splits_ok(){ mem "$BASE/api/dataset/analyze?slug=$1&refresh=1" | $PY -c "import json,sys;d=json.load(sys.stdin);print('yes' if d.get('splits',{}).get('train') and len(d['annotations']['classes'])==2 else 'no')"; }

echo "== 2. upload via every path =="
S_ZIP=$(up_raw /tmp/e2e/bal.zip zipbal); ck "zip classification upload" yes "$([ -n "$S_ZIP" ] && echo yes||echo no)"
ck "  zip splits+2classes (no double-nest)" yes "$(splits_ok "$S_ZIP")"
S_TGZ=$(up_raw /tmp/e2e/bal.tgz tgzbal); ck "tar.gz upload" yes "$(splits_ok "$S_TGZ")"
S_IMG=$(up_raw /tmp/e2e/single.png single); ck "single image upload (1 img)" 1 "$(mem "$BASE/api/dataset/analyze?slug=$S_IMG&refresh=1" | jget "d['n_images']")"
S_LOOSE=$(mem --max-time 60 -F "files=@/tmp/e2e/l1.png" -F "files=@/tmp/e2e/l2.png" -F "files=@/tmp/e2e/l3.png" "$BASE/api/dataset/upload?domain=e2e_ds&name=loose" | jget "d.get('slug','')"); ck "multiple loose files (3 img)" 3 "$(mem "$BASE/api/dataset/analyze?slug=$S_LOOSE&refresh=1" | jget "d['n_images']")"
S_FOLD=$(mem --max-time 60 -F "files=@/tmp/e2e/bal/images/train/catA/catA_train_0.png;filename=ds/train/catA/a.png" -F "files=@/tmp/e2e/bal/images/val/catB/catB_val_0.png;filename=ds/val/catB/b.png" "$BASE/api/dataset/upload?domain=e2e_ds&name=folder" | jget "d.get('slug','')"); ck "folder upload preserves split" yes "$(mem "$BASE/api/dataset/analyze?slug=$S_FOLD&refresh=1" | $PY -c "import json,sys;print('yes' if json.load(sys.stdin).get('splits',{}).get('train') else 'no')")"
S_GOAL=$(up_raw /tmp/e2e/bal.zip withgoal "&goal=classify%20catA%20vs%20catB%20for%20a%20demo"); ck "goal persisted" yes "$(mem "$BASE/api/dataset/uploads?domain=e2e_ds" | $PY -c "import json,sys;print('yes' if any(u['slug']=='$S_GOAL' and 'catA' in (u.get('goal') or '') for u in json.load(sys.stdin)['uploads']) else 'no')")"

echo "== 3. EDA analysis =="
ck "analyze ok + dims + classes" yes "$(mem "$BASE/api/dataset/analyze?slug=$S_ZIP" | $PY -c "import json,sys;d=json.load(sys.stdin);print('yes' if d['ok'] and d['images'].get('ok') and d['annotations']['classes'] else 'no')")"

echo "== 4. AI review + readiness + fitness =="
AIG=$(mem --max-time 90 "$BASE/api/dataset/analyze/ai?slug=$S_GOAL&refresh=1")
ck "ai review ok" True "$(echo "$AIG" | jget "d['ok']")"
ckin "ai source" "ai,rules" "$(echo "$AIG" | jget "d['source']")"
ck "ai has readiness" yes "$(echo "$AIG" | $PY -c "import json,sys;print('yes' if 'suggested_task' in json.load(sys.stdin)['training_readiness'] else 'no')")"
ck "ai has fitness (goal set)" yes "$(echo "$AIG" | $PY -c "import json,sys;print('yes' if json.load(sys.stdin).get('fitness') is not None else 'no')")"

echo "== 5. edge cases =="
AIU=$(mem --max-time 90 "$BASE/api/dataset/analyze/ai?slug=$(up_raw /tmp/e2e/unl.zip unlabeled)&refresh=1")
ck "unlabeled -> not ready" False "$(echo "$AIU" | jget "d['training_readiness']['ready']")"
ck "unlabeled -> no-labels issue" yes "$(echo "$AIU" | $PY -c "import json,sys;d=json.load(sys.stdin);print('yes' if any('label' in (i.get('title','')+i.get('detail','')).lower() for i in d['issues']) else 'no')")"
AII=$(mem --max-time 90 "$BASE/api/dataset/analyze/ai?slug=$(up_raw /tmp/e2e/imb.zip imbalanced)&refresh=1")
ck "imbalanced -> imbalance issue" yes "$(echo "$AII" | $PY -c "import json,sys;d=json.load(sys.stdin);print('yes' if any('imbalance' in i.get('title','').lower() for i in d['issues']) else 'no')")"
S_SEN=$(up_raw /tmp/e2e/sen.tgz sensor "&modality=sensor"); ck "non-image (sensor) analyze" yes "$(mem "$BASE/api/dataset/analyze?slug=$S_SEN&refresh=1" | $PY -c "import json,sys;d=json.load(sys.stdin);md=d.get('modality_detail',{});print('yes' if ('sensor' in md or 'text' in md) else 'no')")"
ck "bad upload (text) -> 400" 400 "$(HC -H "$CK" -X POST -H 'Content-Type: application/octet-stream' --data-binary @/tmp/e2e/bad.txt "$BASE/api/dataset/upload?domain=e2e_ds&name=bad")"
ck "empty upload -> 400" 400 "$(printf '' | HC -H "$CK" -X POST -H 'Content-Type: application/octet-stream' --data-binary @- "$BASE/api/dataset/upload?domain=e2e_ds&name=empty")"

echo "== 6. voice transcribe =="
$PY -c "import wave,struct,math;w=wave.open('/tmp/e2e/v.wav','w');w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(b''.join(struct.pack('<h',int(800*math.sin(2*math.pi*240*t/16000))) for t in range(8000)));w.close()"
ck "voice transcribe ok+text" yes "$(mem --max-time 60 -X POST -H 'Content-Type: application/octet-stream' --data-binary @/tmp/e2e/v.wav "$BASE/api/voice/transcribe" | $PY -c "import json,sys;d=json.load(sys.stdin);print('yes' if d.get('ok') and 'text' in d else 'no')")"

echo "== 7. permission gate + training =="
ck "member train BEFORE grant -> 403" 403 "$(HC -H "$CK" -X POST -H 'Content-Type: application/json' -d "{\"domain\":\"e2e_ds\",\"slug\":\"$S_ZIP\",\"epochs\":1,\"task\":\"classification\"}" "$BASE/api/train/submit")"
curl -s $BA -X POST -H 'Content-Type: application/json' -d '{"user_id":"e2e@uni.edu","allow":true}' "$BASE/api/users/cluster_access" >/dev/null
TR=$(mem --max-time 200 -X POST -H 'Content-Type: application/json' -d "{\"domain\":\"e2e_ds\",\"slug\":\"$S_ZIP\",\"epochs\":1,\"task\":\"classification\"}" "$BASE/api/train/submit")
ck "member train AFTER grant -> job" yes "$(echo "$TR" | $PY -c "import json,sys;d=json.load(sys.stdin);print('yes' if d.get('ok') and ('Submitted batch job' in (d.get('msg') or '')) else 'no')")"

echo "== 8. cleanup =="
curl -s $BA -X POST -H 'Content-Type: application/json' -d '{"user_id":"e2e@uni.edu","allow":false}' "$BASE/api/users/cluster_access" >/dev/null
ck "delete project (cascade)" 200 "$(HC -H "$CK" -X POST -H 'Content-Type: application/json' -d '{"domain":"e2e_ds"}' "$BASE/api/agent/delete")"
rm -rf /tmp/e2e

echo ""; echo "E2E RESULT: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
