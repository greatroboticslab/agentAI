import json, os
from weed_optimizer_framework.tools import db
repo = os.environ.get("REPO_ROOT", os.path.expanduser("~/weed_llm_benchmark"))
reg = json.load(open(os.path.join(repo, "results/framework/dataset_registry.json")))
res = db.mirror_registry_to_mongo(reg)
print("mirror result:", res)
dbh = db._get_db()
for c in ["slugs", "classes", "domains", "images"]:
    try: print(f"  {c}: {dbh[c].count_documents({})}")
    except Exception as e: print(f"  {c}: err {e}")
