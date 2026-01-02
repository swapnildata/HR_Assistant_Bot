- This is langgraph based chatbot created for employees of company. The user can ask the question about HR policies, his/her HRMS data and also create a text file including escalation mail to HR.

Chatbot consists of following three tools:
- HR_chatbot:- A vector Database is created using chroma DB library for HR policies from pdf file. With help of this DB, answers can be retrieved.
- get_employee_hrms_tool:- This tool is fetching the HRMS data from remote MCP server tool.
- Escalation_tool:- This tool will create the .text file which will have a escalation mail of issue faced by employee.

- Using a decorator called 'traceable', a langsmith observebility is added for debugging, and LLMOps purpose.
- The langgraph also has the human-in-the-loop functionality using 'inturrupt' keyword.
- streaming is implemented for better user experience.
- You can visualise the graph in 'graph_visual.ipynb'.