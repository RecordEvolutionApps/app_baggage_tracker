# 1. Set a base image
FROM python:3.9-slim

# 2. (optional) use python wheels from piwheels.org (speeds up build time for arm architectures)
RUN echo '[global]' > /etc/pip.conf && echo 'extra-index-url=https://www.piwheels.org/simple' >> /etc/pip.conf

# 3. Copy over a list of required dependencies and install them
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 4. Create a main app directory and copy over the code
RUN mkdir app
WORKDIR /app
COPY index.py app/index.py

# 5. Start the application
CMD ["python", "-u", "app/index.py"]