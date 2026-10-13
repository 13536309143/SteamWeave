FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY sg_agents ./sg_agents
COPY static ./static
COPY research ./research
EXPOSE 8787
CMD ["python", "-m", "uvicorn", "sg_agents.app:app", "--host", "0.0.0.0", "--port", "8787"]
