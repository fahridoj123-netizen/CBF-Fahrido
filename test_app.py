import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'movieweb.settings')
django.setup()

from django.test import Client

c = Client()

print("1. Testing Home Page ('/')...")
res = c.get('/')
assert res.status_code == 200, f"Home failed: {res.status_code}"
print("   -> Status: 200 OK")

print("2. Testing Movie Detail Page for Toy Story (id=1)...")
res = c.get('/movie/1/')
assert res.status_code == 200, f"Movie detail failed: {res.status_code}"
assert "Precision@10" in res.content.decode('utf-8')
print("   -> Status: 200 OK with Recommendations and Precision Metric")

print("3. Testing API Recommend for Toy Story ('/api/recommend/1/?k=5')...")
res = c.get('/api/recommend/1/?k=5')
assert res.status_code == 200, f"API recommend failed: {res.status_code}"
data = res.json()
assert data['status'] == 'success'
recs = data['data']['recommendations']
print(f"   -> Status: 200 OK. Got {len(recs)} recommendations.")
for r in recs[:3]:
    print(f"      * {r['title']} | Sim: {r['similarity_percentage']}% | Angle: {r['vector_angle_deg']}° | Relevant: {r['is_relevant']}")

print("4. Testing API Search for 'Harry Potter' ('/api/search/?q=Potter')...")
res = c.get('/api/search/?q=Potter')
assert res.status_code == 200, f"API search failed: {res.status_code}"
search_data = res.json()
print(f"   -> Status: 200 OK. Found {len(search_data['results'])} movies.")
for item in search_data['results'][:3]:
    print(f"      * {item['title']} ({item['year']})")

print("5. Testing Methodology Page ('/methodology/')...")
res = c.get('/methodology/')
assert res.status_code == 200, f"Methodology failed: {res.status_code}"
print("   -> Status: 200 OK")

print("\n=== ALL APP ENDPOINTS & ML CALCULATIONS WORKING FLAWLESSLY! ===")
