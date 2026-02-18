from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
import uvicorn

app = FastAPI()

@app.post("/")
async def catch_post(request: Request):
    # Full headers
    print("=== HEADERS ===")
    print(dict(request.headers))
    
    # Raw body as bytes
    body_bytes = await request.body()
    print("\n=== RAW BODY (bytes) ===")
    print(body_bytes)
    
    # As string
    body_str = body_bytes.decode('utf-8', errors='ignore')
    print("\n=== RAW BODY (str) ===")
    print(body_str)
    
    # Try JSON
    try:
        body_json = json.loads(body_str)
        print("\n=== PARSED JSON ===")
        print(json.dumps(body_json, indent=2))
    except json.JSONDecodeError:
        print("\n=== NOT VALID JSON ===")
    
    print("\n" + "="*50 + "\n")
    
    return JSONResponse({"status": "received", "body_preview": body_str[:200]})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
