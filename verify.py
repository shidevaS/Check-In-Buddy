import requests

base = "http://127.0.0.1:8000"

# Health endpoint
r = requests.get(f"{base}/health")
data = r.json()
print(f"Health: {r.status_code} - provider={data.get('provider')}, model={data.get('model')}")

# Chat/test endpoint
r = requests.post(f"{base}/api/chat/test", json={"user_id": 1, "message": "Good morning!"})
data = r.json()
print(f"Chat/test: {r.status_code} - reply={data.get('reply', 'N/A')}, provider={data.get('provider')}")

# Dashboard API
r = requests.get(f"{base}/api/dashboard/1")
data = r.json()
print(f"Dashboard: {r.status_code} - user={data.get('user', {}).get('name', 'N/A')}, alerts={len(data.get('alerts', []))}")

# Family API
r = requests.get(f"{base}/api/family/1")
data = r.json()
print(f"Family: {r.status_code} - user={data.get('user', {}).get('name', 'N/A')}")

# Senior HTML page
r = requests.get(f"{base}/")
print(f"Senior page: {r.status_code}" + (" - OK" if r.status_code == 200 else ""))

# Family HTML page
r = requests.get(f"{base}/family")
print(f"Family page: {r.status_code}" + (" - OK" if r.status_code == 200 else ""))

# Seed endpoint
r = requests.post(f"{base}/api/seed", json={"variant": "demo", "force": True})
data = r.json()
print(f"Seed: {r.status_code} - status={data.get('status', 'N/A')}, alerts={data.get('alerts', 0)}")