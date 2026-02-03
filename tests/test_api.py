import requests

def test_api_health():
    """Test API health endpoint"""
    response = requests.get("http://localhost:8500/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✓ Health check passed")

def test_api_root():
    """Test API root endpoint"""
    response = requests.get("http://localhost:8500/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    print("✓ Root endpoint passed")

def run_tests():
    """Run all tests"""
    print("Running API tests...")
    try:
        test_api_health()
        test_api_root()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    run_tests()
