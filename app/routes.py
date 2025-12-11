# app/routes.py
from fastapi import APIRouter
from app.services.main import generate_answer

router = APIRouter()


@router.post("/get-answer")
def process(data: dict):
    query = data.get("query")

    result = generate_answer(query)
    return result
