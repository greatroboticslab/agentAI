#!/usr/bin/env bash
# ===========================================================================
# AgentAI dashboard smoke test (v3.0.136) — repeatable, self-cleaning.
#
# A committed, professional replacement for ad-hoc curl checks: exercises every
# page, API, and key user flow, prints a PASS/FAIL report, and exits non-zero on
# any failure (CI-friendly). Run it after every deploy.
#
#   bash tests/smoke_test.sh                      # localhost, Basic 1/1
#   BASE=https://lab-b660m-c.tailfa6424.ts.net bash tests/smoke_test.sh
#   DASH_USER=1 DASH_PASS=1 bash tests/smoke_test.sh
#
# RBAC/session-cookie checks run only when ~/.dash_session_key is readable
# (i.e. on the lab host); elsewhere they are skipped (reported as SKIP).
# ===========================================================================
BASE="${BASE:-http://127.0.0.1:8000}"
U="${DASH_USER:-1}"; P="${DASH_PASS:-1}"
pass=0; fail=0; skip=0
ck(){ if [ "$3" = "$2" ]; then pass=$((pass+1)); printf "  ok    %-44s %s\n" "$1" "$3";
      else fail=$((fail+1)); printf "  FAIL  %-44s got=%s exp=%s\n" "$1" "$3" "$2"; fi; }
ckin(){ # pass if $3 is in the set $2 (comma-separated)
  case ",$2," in *,"$3",*) pass=$((pass+1)); printf "  ok    %-44s %s\n" "$1" "$3";;
  *) fail=$((fail+1)); printf "  FAIL  %-44s got=%s exp_one_of=%s\n" "$1" "$3" "$2";; esac; }
HC(){ curl -s -o /dev/null -w '%{http_code}' --max-time 40 "$@"; }
BA="-u $U:$P"

echo "AgentAI smoke test → $BASE"
CLS=$(curl -s $BA --max-time 25 "$BASE/api/annotation_status" | python3 -c "import json,sys;d=json.load(sys.stdin);print((d['rows'][0]['cwd12'] or ['Crabgrass'])[0])" 2>/dev/null || echo Crabgrass)
SLUG=$(curl -s $BA --max-time 25 "$BASE/api/annotation_status" | python3 -c "import json,sys;print(json.load(sys.stdin)['rows'][0]['slug'])" 2>/dev/null || echo cottonweed_sp8)
echo "(class=$CLS slug=$SLUG)"

echo "== PAGES =="
for pg in / /agent/weed /agent/mobile_robot /agent/humanoid_robot /classes "/classes/$CLS" /slugs "/gallery/$SLUG" /labeling /annotate /roboflow /rounds /users /login /console; do
  ck "GET $pg" 200 "$(HC $BA "$BASE$pg")"
done

echo "== APIs =="
for api in /api/me /api/users /api/annotation_status /api/rounds_state "/api/domain/push_cap?domain=weed" "/api/dataset/uploads?domain=weed" /api/cluster_status /api/roboflow_status /healthz; do
  ck "GET $api" 200 "$(HC $BA "$BASE$api")"
done

echo "== DATASET ANALYSIS (EDA) =="
ck "GET /dataset/$SLUG" 200 "$(HC $BA "$BASE/dataset/$SLUG")"
AN=$(curl -s $BA --max-time 90 "$BASE/api/dataset/analyze?slug=$SLUG")
ck "analyze ok" True "$(echo "$AN" | python3 -c "import json,sys;print(json.load(sys.stdin)['ok'])" 2>/dev/null)"
ck "analyze has class distribution" yes "$(echo "$AN" | python3 -c "import json,sys;a=json.load(sys.stdin)['annotations'];print('yes' if a['classes'] else 'no')" 2>/dev/null)"
ck "analyze has image dim stats" yes "$(echo "$AN" | python3 -c "import json,sys;print('yes' if json.load(sys.stdin)['images'].get('ok') else 'no')" 2>/dev/null)"

echo "== AUTH =="
ck   "basic creds -> 200"        200       "$(HC $BA "$BASE/agent/weed")"
ckin "no-cred -> login/401"      "302,401" "$(HC -H 'Accept: text/html' "$BASE/")"
ck   "forged cookie -> 401"      401       "$(HC -H 'Cookie: agentai_session=x.y' "$BASE/agent/weed")"

echo "== UPLOAD / LIST / DELETE =="
TMP=$(mktemp -d); mkdir -p "$TMP/images"
python3 -c "import base64;from pathlib import Path;Path('$TMP/images/a.png').write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBAQDJ/1eqAAAAAElFTkSuQmCC'))"
( cd "$TMP" && zip -qr d.zip images )
UP=$(curl -s $BA --max-time 60 -X POST -H 'Content-Type: application/zip' --data-binary @"$TMP/d.zip" "$BASE/api/dataset/upload?domain=mobile_robot&name=smoke")
USLUG=$(echo "$UP" | python3 -c "import json,sys;print(json.load(sys.stdin).get('slug',''))" 2>/dev/null)
ck "upload returns slug" yes "$([ -n "$USLUG" ] && echo yes || echo no)"
ck "upload listed" yes "$(curl -s $BA --max-time 25 "$BASE/api/dataset/uploads?domain=mobile_robot" | python3 -c "import json,sys;print('yes' if any(u['slug']=='$USLUG' for u in json.load(sys.stdin)['uploads']) else 'no')" 2>/dev/null)"
ck "delete own upload -> 200" 200 "$(HC $BA -X POST -H 'Content-Type: application/json' -d "{\"slug\":\"$USLUG\"}" "$BASE/api/dataset/delete")"
ck "delete harvested -> 403" 403 "$(HC $BA -X POST -H 'Content-Type: application/json' -d '{"slug":"cottonweed_sp8"}' "$BASE/api/dataset/delete")"
rm -rf "$TMP"

echo "== PUSH CAP =="
OLD=$(curl -s $BA --max-time 15 "$BASE/api/domain/push_cap?domain=weed" | python3 -c "import json,sys;print(json.load(sys.stdin)['cap'])" 2>/dev/null)
ck "set cap=88 -> 200" 200 "$(HC $BA -X POST -H 'Content-Type: application/json' -d '{"domain":"weed","cap":88}' "$BASE/api/domain/push_cap")"
ck "get cap=88" 88 "$(curl -s $BA --max-time 15 "$BASE/api/domain/push_cap?domain=weed" | python3 -c "import json,sys;print(json.load(sys.stdin)['cap'])" 2>/dev/null)"
curl -s $BA -X POST -H 'Content-Type: application/json' -d "{\"domain\":\"weed\",\"cap\":${OLD:-100}}" "$BASE/api/domain/push_cap" >/dev/null

echo "== RBAC (session cookies) =="
KEY="$HOME/.dash_session_key"
if [ -r "$KEY" ]; then
  mkc(){ python3 - "$1" "$KEY" <<PY
import base64,hmac,hashlib,json,time,sys
key=open(sys.argv[2],'rb').read()
b=lambda x:base64.urlsafe_b64encode(x).decode().rstrip('=')
p={'uid':sys.argv[1],'email':sys.argv[1],'name':sys.argv[1],'exp':time.time()+3600}
body=b(json.dumps(p,separators=(',',':')).encode())
print(body+'.'+b(hmac.new(key,body.encode(),hashlib.sha256).digest()))
PY
}
  M=$(mkc smoke_member@example.com)
  M2=$(mkc smoke_other@example.com)
  ck "member train -> 403"        403  "$(HC -H "Cookie: agentai_session=$M" -X POST "$BASE/api/cluster_action/clean_train_d")"
  ck "member set_role -> 403"     403  "$(HC -H "Cookie: agentai_session=$M" -X POST -H 'Content-Type: application/json' -d '{"user_id":"x","role":"admin"}' "$BASE/api/users/role")"
  ck "member cluster_access ->403" 403 "$(HC -H "Cookie: agentai_session=$M" -X POST -H 'Content-Type: application/json' -d '{"user_id":"x","allow":true}' "$BASE/api/users/cluster_access")"
  ck "member can_use_cluster=false" false "$(curl -s -H "Cookie: agentai_session=$M" --max-time 15 "$BASE/api/me" | python3 -c "import json,sys;print(str(json.load(sys.stdin)['can_use_cluster']).lower())" 2>/dev/null)"

  echo "== PROJECT / AGENT lifecycle (member) =="
  HC -H "Cookie: agentai_session=$M" -X POST -H 'Content-Type: application/json' -d '{"domain":"smoke_proj"}' "$BASE/api/agent/delete" >/dev/null  # clean stale
  ck "member create project" 200 "$(HC -H "Cookie: agentai_session=$M" -X POST -H 'Content-Type: application/json' -d '{"name":"smoke proj","research_field":"testing","modality":["image"]}' "$BASE/api/agent/create")"
  ck "add agent (collector)" 200 "$(HC -H "Cookie: agentai_session=$M" -X POST -H 'Content-Type: application/json' -d '{"project":"smoke_proj","type":"collector"}' "$BASE/api/project/agent/add")"
  ck "agents list = 1" 1 "$(curl -s -H "Cookie: agentai_session=$M" --max-time 15 "$BASE/api/project/agents?project=smoke_proj" | python3 -c "import json,sys;print(len(json.load(sys.stdin)['agents']))" 2>/dev/null)"
  AID=$(curl -s -H "Cookie: agentai_session=$M" --max-time 15 "$BASE/api/project/agents?project=smoke_proj" | python3 -c "import json,sys;a=json.load(sys.stdin)['agents'];print(a[0]['id'] if a else '')" 2>/dev/null)
  ck "non-owner add agent -> 403" 403 "$(HC -H "Cookie: agentai_session=$M2" -X POST -H 'Content-Type: application/json' -d '{"project":"smoke_proj","type":"filter"}' "$BASE/api/project/agent/add")"
  ck "owner remove agent" 200 "$(HC -H "Cookie: agentai_session=$M" -X POST -H 'Content-Type: application/json' -d "{\"project\":\"smoke_proj\",\"agent_id\":\"$AID\"}" "$BASE/api/project/agent/remove")"
  ck "agents list = 0" 0 "$(curl -s -H "Cookie: agentai_session=$M" --max-time 15 "$BASE/api/project/agents?project=smoke_proj" | python3 -c "import json,sys;print(len(json.load(sys.stdin)['agents']))" 2>/dev/null)"
  # upload a dataset to the project, then delete project should cascade it away
  TMP2=$(mktemp -d); mkdir -p "$TMP2/images"
  python3 -c "import base64;from pathlib import Path;Path('$TMP2/images/a.png').write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M8AAAMBAQDJ/1eqAAAAAElFTkSuQmCC'))"
  ( cd "$TMP2" && zip -qr d.zip images )
  PSLUG=$(curl -s -H "Cookie: agentai_session=$M" --max-time 40 -X POST -H 'Content-Type: application/zip' --data-binary @"$TMP2/d.zip" "$BASE/api/dataset/upload?domain=smoke_proj&name=smokeds" | python3 -c "import json,sys;print(json.load(sys.stdin).get('slug',''))" 2>/dev/null)
  ck "project upload listed" yes "$(curl -s -H "Cookie: agentai_session=$M" --max-time 20 "$BASE/api/dataset/uploads?domain=smoke_proj" | python3 -c "import json,sys;print('yes' if json.load(sys.stdin)['uploads'] else 'no')" 2>/dev/null)"
  ck "non-owner delete project -> 403" 403 "$(HC -H "Cookie: agentai_session=$M2" -X POST -H 'Content-Type: application/json' -d '{"domain":"smoke_proj"}' "$BASE/api/agent/delete")"
  ck "owner delete project (cascade)" 200 "$(HC -H "Cookie: agentai_session=$M" -X POST -H 'Content-Type: application/json' -d '{"domain":"smoke_proj"}' "$BASE/api/agent/delete")"
  ck "project datasets gone after delete" 0 "$(curl -s -H "Cookie: agentai_session=$M" --max-time 20 "$BASE/api/dataset/uploads?domain=smoke_proj" | python3 -c "import json,sys;print(len(json.load(sys.stdin)['uploads']))" 2>/dev/null)"
  rm -rf "$TMP2"

  echo "== MODEL CATALOG (deployed-only) =="
  ck "catalog 200" 200 "$(HC $BA "$BASE/api/models/catalog")"
  ck "catalog has yolo11s" yes "$(curl -s $BA --max-time 15 "$BASE/api/models/catalog" | python3 -c "import json,sys;print('yes' if any(m['id']=='yolo11s' for m in json.load(sys.stdin)['models']) else 'no')" 2>/dev/null)"
  ck "catalog has deployed gemma4 brain" yes "$(curl -s $BA --max-time 15 "$BASE/api/models/catalog" | python3 -c "import json,sys;print('yes' if any(m['id']=='ollama:gemma4' for m in json.load(sys.stdin)['models']) else 'no')" 2>/dev/null)"
  ck "catalog excludes cloud giants" yes "$(curl -s $BA --max-time 15 "$BASE/api/models/catalog" | python3 -c "import json,sys;ids=[m['id'] for m in json.load(sys.stdin)['models']];print('yes' if not any(':cloud' in i or 'v4-pro' in i or 'glm-5' in i for i in ids) else 'no')" 2>/dev/null)"
  ck "deploy rejects cloud model" 400 "$(HC $BA -X POST -H 'Content-Type: application/json' -d '{"model":"deepseek-v4-pro"}' "$BASE/api/models/deploy")"
  ck "member deploy -> 403" 403 "$(HC -H "Cookie: agentai_session=$M" -X POST -H 'Content-Type: application/json' -d '{"model":"deepseek-v4"}' "$BASE/api/models/deploy")"

  echo "== INTENT -> PLAN (agent-builder V2) =="
  ck "plan empty desc -> 400" 400 "$(HC -H "Cookie: agentai_session=$M" -X POST -H 'Content-Type: application/json' -d '{"description":""}' "$BASE/api/agent/plan")"
  PLAN=$(curl -s -H "Cookie: agentai_session=$M" -X POST -H 'Content-Type: application/json' -d '{"description":"collect drone images and train a model to detect bleaching"}' --max-time 45 "$BASE/api/agent/plan")
  ck "plan ok" True "$(echo "$PLAN" | python3 -c "import json,sys;print(json.load(sys.stdin)['ok'])" 2>/dev/null)"
  ckin "plan source ai/heuristic" "ai,heuristic" "$(echo "$PLAN" | python3 -c "import json,sys;print(json.load(sys.stdin)['source'])" 2>/dev/null)"
  ck "plan proposes >=1 agent" yes "$(echo "$PLAN" | python3 -c "import json,sys;print('yes' if json.load(sys.stdin)['plan']['agents'] else 'no')" 2>/dev/null)"
  ck "plan agent types valid" yes "$(echo "$PLAN" | python3 -c "import json,sys;V={'collector','filter','labeler','trainer','evaluator','custom'};a=json.load(sys.stdin)['plan']['agents'];print('yes' if all(x['type'] in V for x in a) else 'no')" 2>/dev/null)"
else
  skip=$((skip+23)); echo "  SKIP  RBAC + project/agent + catalog + plan cookie checks (no ~/.dash_session_key on this host)"
fi

echo ""
echo "RESULT: $pass passed, $fail failed, $skip skipped"
[ "$fail" -eq 0 ]
