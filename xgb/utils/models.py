from pydantic import BaseModel, Field


class XGBScore(BaseModel):
    accuracy: float = Field(..., description="Accuracy of the model")
    macro_f1: float = Field(..., description="Macro F1 score of the model")
    macro_precision: float = Field(
        ..., description="Macro precision score of the model"
    )
    macro_recall: float = Field(..., description="Macro recall score of the model")
