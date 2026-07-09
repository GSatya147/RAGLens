from litellm import (
    completion, 
    RateLimitError, 
    APIError, 
    ServiceUnavailableError, 
    Timeout, 
    AuthenticationError, 
    BadRequestError
)

from configs.config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL

class Client:
    def __init__(self):
        pass

    def generate_response(self, messages):
        try:
            response = completion(
                api_key=DEEPSEEK_API_KEY,
                model=DEEPSEEK_MODEL,
                messages=messages,
                # max_tokens=500,
                num_retries=3
            )

            message = response.choices[0].message.model_dump()
            return message
    
        except RateLimitError as e:
                print(f"Rate limit error: {e.message}")

        except APIError as e:
                print(f"API error: {e.message} ")

        except BadRequestError as e:
                print(f"Bad request error: {e.message} ")

        except Timeout as e:
                print(f"Time out error: {e.message} ")
            
        except AuthenticationError as e:
                print(f"Auth error: {e.message} ")

        except ServiceUnavailableError as e:
                print(f"Service unavailable error: {e.message} ")

        except Exception as e:
                print(f"Unexpected error: {e}")

if __name__=="__main__":
       obj = Client()
       result = obj.generate_response([{"role": "system", "content" : "You are a friendly english speaking user assistant"}, {"role": "user", "content" : "Hello"}])
       print(result)
