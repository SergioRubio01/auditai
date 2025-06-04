from pydantic import BaseModel
from datetime import datetime

class ImageResponse(BaseModel):
    filename: str
    path: str
    size: int
    last_modified: datetime

    class Config:
        from_attributes = True  # For Pydantic v2