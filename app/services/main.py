from app.services.database import vector_store
from app.services.translator import local_translate
from app.services.chatbot import provide_answer

SUCCESS_STATUS = {"CODE": 200, "MESSAGE": "OK"}

FAILED_STATUS = {"CODE": 404, "MESSAGE": "NOT_OK"}


def generate_answer(query):

    result = {"status": "", "data": []}

    try:
        nearest_answers = []
        english_query = local_translate(query)
        answers_results = vector_store.similarity_search(query=english_query, k=5)
        for answer in answers_results:
            res = {"answer": "", "metadata": {}}
            res["answer"] = answer.metadata["farsi-text"]
            res["metadata"] = answer.metadata
            res["eng-text"] = answer.page_content
            nearest_answers.append(res)
        # answer = provide_answer(query, nearest_answers)

        result["status"] = SUCCESS_STATUS["MESSAGE"]
        result["data"] = nearest_answers

        return result
    except Exception as e:
        result["status"] = FAILED_STATUS["MESSAGE"]
        result["data"] = "There is an issue from our side please be patience"
        print(e)
        return result
