from app.services.database import vector_store

SUCCESS_STATUS = {"CODE": 200, "MESSAGE": "OK"}

FAILED_STATUS = {"CODE": 404, "MESSAGE": "NOT_OK"}


def generate_answer(query):

    result = {"status": "", "answer": "", "metadata": {}}

    try:
        answers_results = vector_store.similarity_search(query=query, k=1)
        answer = answers_results[0]
        result["status"] = SUCCESS_STATUS["MESSAGE"]
        result["answer"] = answer.page_content
        result["metadata"] = answer.metadata

        return result
    except:
        result["status"] = FAILED_STATUS["MESSAGE"]
        result["answer"] = "There is an issue from our side please be patience"
        return result
