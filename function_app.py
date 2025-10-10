import azure.functions as func

from tvbingefriend_recommendation_service.blueprints import recommendations_bp

app = func.FunctionApp()

app.register_blueprint(recommendations_bp)
