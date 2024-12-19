from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    open_api_key: str

    class Config:
        env_file = ".env"


setting = Settings()  # pyright: ignore

if __name__ == "__main__":
    print(setting)
