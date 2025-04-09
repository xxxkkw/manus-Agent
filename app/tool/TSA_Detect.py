from app.tool.base import BaseTool, ToolResult  
from TSA.detect import detect
import json


class TSA_Detect(BaseTool):
    """
    Sleep Apnea detection tool using Multi-Scaled CNN with SE block (SE-MSCNN) based on single-lead ECG signals. Features:
    - Multi-scale feature extraction: Captures local/global ECG patterns through triple sub-networks
    - Adaptive fusion: Channel-wise attention with Squeeze-and-Excitation block enables smart feature fusion
    - Lightweight design: 89K parameters suitable for wearable device deployment
    - High accuracy: Superior performance on Apnea-ECG benchmark
    Clinical applications: Home Sleep Test (HST) for early warning of cardiovascular diseases, diabetes and sudden death prevention.
    """
    name: str = "TSA_Detect"
    description: str = """
    This tool, based on a lightweight multi-scaled fusion network, is designed for sleep apnea detection. 
    Use this tool when you need to analyze sleep apnea data. It generates detailed data for each time point 
    indicating whether sleep apnea occurred during the measurement period, along with visualizations. 
    For each subject, you must generate a comprehensive report including:
    - Analyze their current sleep quality
    - Evaluate sleep apnea occurrences during their sleep
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
                    'average_rate': float(result[-2].split(": ")[1][:-1])/100,
                    'max_risk': result[-3].split("(")[0].strip(),
                    'output_dir': result[-1].split(": ")[1]
                }
                return ToolResult(
                    output=json.dumps({
                        "report": result,
                        "statistics": stats
                    }, ensure_ascii=False, indent=2)
                )
            else:
                return ToolResult(output='\n'.join(result))
                
        except Exception as e:
            return ToolResult(error=f"执行错误: {str(e)}")

    
