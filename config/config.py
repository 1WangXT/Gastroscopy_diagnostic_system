from pydantic import BaseSettings


class Settings(BaseSettings):
    DEBUG: bool = True

    TITLE: str = "pyqt5_example"

    # Mysql
    MYSQL_USERNAME: str = "root"
    MYSQL_PASSWORD: str = "123456"
    MYSQL_HOST: str = "127.0.0.1"
    MYSQL_PORT: int = 3306
    MYSQL_DATABASE: str = 'pyqt5_example_db'


settings = Settings()
