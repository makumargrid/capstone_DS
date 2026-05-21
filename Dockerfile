# Use Miniconda as the base to handle complex C++ dependencies
FROM continuumio/miniconda3:latest

# Set a working directory
WORKDIR /app

# Install CadQuery using the official Conda channels
RUN conda install -y -c cadquery -c conda-forge cadquery

# Install MeshLib, Google GenAI SDK, ADK, and server packages via pip
RUN pip install meshlib google-genai google-adk[db]==1.18.0 sqlalchemy aiosqlite fastapi uvicorn pydantic python-dotenv python-multipart

# Copy your pipeline script into the container
COPY . /app

# The command that runs when the container starts
CMD ["python", "pipeline.py"]