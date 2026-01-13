"""FastAPI application for Zoo Acoustics API"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from api.routers import metadata, activity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Zoo Acoustics API",
    description="""
    API for analyzing acoustic monitoring data from zoo enclosures.
    
    ## Features
    
    * **Metadata Endpoints**: Browse available zoos, enclosures, and dates
    * **Activity Endpoint**: Get detailed nighttime acoustic activity data for visualization
    
    ## Nighttime Window
    
    The nighttime window is defined as **8 PM to 3 AM** (hours 20-23, 0-2).
    When you query for a specific date, you get data from 8 PM on that date to 3 AM the following day.
    
    ## Data Structure
    
    Activity data is binned into **10-minute intervals** and includes:
    1. **Species Activity**: Target bird species detections per bin
    2. **Overall Activity**: Total acoustic activity (all sounds) per bin  
    3. **MIT Breakdown**: Sound classification breakdown per bin (top 8 types)
    
    ## Example Usage
    
    1. Get all zoos: `GET /api/zoos`
    2. Get enclosures: `GET /api/zoos/GaiaZOO/enclosures`
    3. Get available dates: `GET /api/zoos/GaiaZOO/enclosures/Congo/dates`
     4. Get activity data: `GET /api/zoos/GaiaZOO/enclosures/Congo/activity?date=2024-08-10`
     5. Get activity for specific species: `GET /api/zoos/GaiaZOO/enclosures/Congo/activity?date=2024-08-10&species=Quelea%20quelea,Upupa%20epops`
    """,
    version="1.0.0",
    contact={
        "name": "Zoo Acoustics Team"
    }
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(metadata.router)
app.include_router(activity.router)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Zoo Acoustics API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "zoos": "/api/zoos",
            "enclosures": "/api/zoos/{zoo_name}/enclosures",
            "species": "/api/zoos/{zoo_name}/enclosures/{enclosure_name}/species",
            "dates": "/api/zoos/{zoo_name}/enclosures/{enclosure_name}/dates",
            "activity": "/api/zoos/{zoo_name}/enclosures/{enclosure_name}/activity?date=YYYY-MM-DD&species=SCIENTIFIC_NAME[,SCIENTIFIC_NAME]"
        }
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "zoo-acoustics-api"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
