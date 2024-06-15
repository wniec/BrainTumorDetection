from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Prediction(BaseModel):
    # def __init__(self, link, subject, evaluation, danger):
    #     self.id = uuid.uuid4()
    #     self.link = link
    #     self.subject = subject
    #     self.evaluation = evaluation
    #     self.danger = danger
    id: str
    link: str
    subject: str
    evaluation: int
    danger: int


predictions = []


@app.get("/preds", response_model=list[Prediction])
def general_predictions():
    predictions.sort(key=lambda x: x["danger"], reverse=True)
    # out = []
    # for p in predictions:
    #     out.append([p.link, p.danger, p.evaluation, p.id])
    print("Returning: ", predictions)
    return predictions
    # return {"link": "Welcome to the REST API"}


def findByUUID(id: str):
    for p in predictions:
        if p["id"] == id:
            return p
    else:
        return None


@app.get("/pred/{pred_id}", response_model=Prediction)
def specific_predictions(pred_id: str):
    p = findByUUID(pred_id)
    if p is not None:
        return p
    else:
        raise HTTPException(status_code=404, detail="Prediction not found")


if __name__ == "__main__":
    import uvicorn

    # predictions.append(Prediction("aaa.com/no", "Nathan Oakley", 13, 0))
    # predictions.append(Prediction("aaa.com/yes", "Yantair Evan Simpson", 25, 1))
    # predictions.append(Prediction("aaa.com/idk", "", 112, 10))
    predictions.append(
        {"id": str(uuid.uuid4()), "link": "aaa.com/no", "subject": "Nathan Oakley", "evaluation": 13, "danger": 0})
    predictions.append(
        {"id": str(uuid.uuid4()), "link": "aaa.com/yes", "subject": "Yantair Evan Simpson", "evaluation": 24,
         "danger": 1})
    predictions.append(
        {"id": str(uuid.uuid4()), "link": "aaa.com/idk", "subject": "Ivan Dmovsky-Kolchak", "evaluation": 135,
         "danger": 10})

    uvicorn.run(app, host="127.0.0.1", port=8000)
