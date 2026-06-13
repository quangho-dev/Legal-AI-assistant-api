from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routes.userRoutes import router as userRoutes
from src.routes.filesRoutes import router as filesRoutes

# Create FastAPI app
app = FastAPI(
    title="Legal AI assistant API",
    description="Backend API for legal AI assistant",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(userRoutes, prefix="/api/user")
app.include_router(filesRoutes, prefix="/api")

# @app.get("/health")
# async def health_check():
#     return {"status": "healthy", "version": "1.0.0"}