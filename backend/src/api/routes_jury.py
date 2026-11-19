"""Serve a fixed study without exposing operator records on the jury listener."""

import os
from uuid import UUID

from fastapi import APIRouter, HTTPException

from src.api.routes_studies import Vote, challenge, vote

router = APIRouter(prefix="/jury")


def selection():
    try:
        return UUID(os.environ["MRTL_JURY_STUDY"]), os.environ["MRTL_JURY_RUN"]
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            503, "Select a completed study and model on the jury server"
        ) from exc


@router.post("/challenge")
def draw():
    return challenge(*selection())


@router.post("/vote/{token}")
def reveal(token: UUID, request: Vote):
    return vote(*selection(), token, request)
