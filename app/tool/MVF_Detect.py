from app.tool.base import BaseTool, ToolResult
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from MVF.detect import detect
from MVF.model.Utils import *
import json

class MVF_Detect(BaseTool):
    """
    Multi-View Fusion Network for Sleep Stage Classification, based on
    multi-modal physiological signals of electroencephalography (EEG), 
    electrocardiography (ECG), electrooculography (EOG), and electromyography (EMG). 
    """

    name: str = "MVF_Detect" 
    description: str = """
    This tool based on Multi-View Fusion Network for Sleep Stage Classification, 
    which is designed to analyze user's sleep data and generate detailed stage information. 
    Use this tool when you need to analyze sleep data. It generates detailed data for each time point 
    indicating the predicted sleep stage (W, N1, N2, N3, REM) during the measurement period, along with visualizations. 
    For each subject, you must generate a comprehensive report including:
    - Analyze their current sleep quality
    - Evaluate their sleep stage distribution
    - Provide personalized recommendations and improvement plans
    """
    parameters: dict = {
        "type": "object",
        "properties": {
            "output_format": {
                "type": "string",
                "enum": ["text", "json"],
                "default": "text"
            }
        }
    }

    async def execute(self, **kwargs) -> ToolResult:
        output_format = kwargs.get("output_format", "text")
        try:
            result = detect()
            if output_format == "json":
                stats = {
                    'output_dir': result["save_path"],
                    'stage_distribution': {
                        stage: float(percent[:-1])/100 
                        for line in result["report"][2:-1] 
                        for stage, percent in [line.split(": ")]
                    }
                }
                return ToolResult(
                    output=json.dumps({
                        "report": result["report"],
                        "statistics": stats
                    }, ensure_ascii=False, indent=2)
                )
            else:
                return ToolResult(output='\n'.join(result["report"]))
                
        except Exception as e:
            return ToolResult(error=f"执行错误: {str(e)}")

       