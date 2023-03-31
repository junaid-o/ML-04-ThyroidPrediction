# ML-04-ThyroidPrediction

## Step 1: Create Conda Environment

```
conda create -p venv python==3.10 -y
```

## Step 2: Create following
            a. .github/workflows/main.yaml
            b. .dockerignore
            c. .Dockerfile
            d.  app.py
            e.  requirements.txt

## Step 3: Add The content to files
            a. Add CI/CD file for Gihub Actions as main.yaml in .github/workflows/main.yaml
            b. Add the file names are not be added in Docker Image
            c. Add Content to Dockerfile
            d. Creat flask app

## Step 4: Build Docker Image

    *Build Docker Image*
    ```
    docker build -t <image-name>:<tag-nam> <location-of-docker-file for curren directory just add dot (.)>
    ```
    <span style="color:red">Note: Docker Image name must be lowrcase</span>

    *To Check List of Docker Images*
    ```
    docker images
    ```

    *To Run Docker Image*
    ```
    docker run -p 5000:5000 -e PORT=5000 <Image-ID>
    ```

    *To Check Running Containers in docker*
    ```
    docke ps
    ```

    *To Stop Docker Container*

    ```
    docker stop <container_id>
    ```

## Step 5: Poject Structure Creation
    ### Step A: Create Folder With Project Name <ThyroidPrediction> in root directory
        a. Add a init file ```__init__.py```

    ### Step B: Create setup.py file in root directory