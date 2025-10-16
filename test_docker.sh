echo "Testing Docker deployment..."

# Wait for services to start
sleep 10

# Test FastAPI
echo -n "Testing FastAPI... "
if curl -s http://localhost:8000/health | grep -q "ok"; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
fi

# Test Streamlit
echo -n "Testing Streamlit... "
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8501 | grep -q "200"; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
fi

# Test MLflow
echo -n "Testing MLflow... "
if curl -s http://localhost:5000 | grep -q "MLflow"; then
    echo "✅ PASS"
else
    echo "❌ FAIL"
fi

echo "Done!"
