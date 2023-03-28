FROM python:3.7.5-slim
WORKDIR /app  # first set the working directory
# copy requirements.txt to current working directory
COPY requirements.txt ./  
# install required pkgs
RUN pip install -r requirements.txt
# copy other code
COPY . ./


CMD ["python", "ML-deploy.py"]
