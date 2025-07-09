# app/main.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import base64
from io import BytesIO
from dotenv import load_dotenv

from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from src.models.llms import load_llm
from src.utils import execute_plt_code
from src.logger.base import BaseLogger

load_dotenv()
app = FastAPI()
logger = BaseLogger()
MODEL_NAME = "gemini-2.5-pro"

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Bạn nên giới hạn khi lên production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = load_llm(model_name=MODEL_NAME)

@app.post("/analyze/")
async def analyze_data(file: UploadFile = File(...), query: str = Form(...)):
    try:
        df = pd.read_csv(file.file)
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            agent_type="tool-calling",
            allow_dangerous_code=True,
            verbose=True,
            return_intermediate_steps=True,
        )
        response = agent(query)
        output_text = response.get("output", "")
        intermediate_steps = response.get("intermediate_steps", [])

        code = ""
        image_base64 = None

        if intermediate_steps:
            try:
                tool_call = intermediate_steps[-1][0]
                tool_input = tool_call.tool_input
                if isinstance(tool_input, dict):
                    for val in tool_input.values():
                        if isinstance(val, str) and "plt" in val:
                            code = val
                            break
                if code:
                    fig = execute_plt_code(code, df=df)
                    if fig:
                        buf = BytesIO()
                        fig.savefig(buf, format="png")
                        buf.seek(0)
                        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
            except Exception as e:
                logger.warning(f"Tool parsing error: {e}")

        return JSONResponse(content={
            "output": output_text,
            "code": code,
            "image_base64": image_base64
        })

    except Exception as e:
        logger.error(f"Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
