import os
import shutil
from pathlib import Path
import cv2
import numpy
from flask import Flask, render_template, request, redirect, url_for, json, send_file, flash
from app.repositories import models_repository, results_repository
from app.services.images import load_image_by_name
from app.services.mapbox import store_composite, delete_composite, get_all_mapboxes, get_mapbox_by_id
from app.services.models.models import analyze_image, store_model, filter_results

ANALYZED_IMAGES_PATH = os.environ.get('ANALYZED_PATH')


def validate(params):
    errors = {}
    title = params.get('title', '')
    description = params.get('description', '')
    coordinates = params.get('coordinates', '')

    if len(title) == 0:
        errors['title'] = 'Title can not be empty'

    if len(description) == 0:
        errors['description'] = 'Description can not be empty'

    if len(coordinates) == 0:
        errors['coordinates'] = 'Coordinates can not be empty'
    elif len(coordinates.split(';')) != 4:
        errors['coordinates'] = 'Incorrect format'

    return errors


def create_app():
    app = Flask(__name__)

    environment_configuration = os.environ.get('CONFIGURATION_SETUP')
    app.secret_key = os.environ.get('SECRET_KEY')
    app.config.from_object(environment_configuration)

    import db
    db.init_app(app)

    # register blueprint
    # app.register_blueprint(main)
    # app.config.from_pyfile('config.py')

    @app.route("/")
    def index():
        response = get_all_mapboxes()
        models = models_repository.get_all()
        mapboxes = [tuple(row) for row in response]
        return render_template('index.html', mapboxes=json.dumps(mapboxes, default=str), models=models)

    @app.route("/filtering", methods=('GET', 'POST'))
    def filtering_index():
        if request.method == 'GET':
            useful_models = [
                'model_120_no_optim_slope_best.pt',
                'model_200_no_optim_msrm_x1_best.pt',
                'model_200_no_optim_msrm_x2_best.pt',
                'model_200_no_optim_slrm_best.pt'
            ]
            results = results_repository.get_all()
            results = [tuple(row) for row in results]
            results = [row for row in results if row[6] in useful_models]
            return render_template('filtering/index.html', results=results)
        data = request.get_json()
        result_path = filter_results(data)
        flash('Results was successfully filtered and new results are stored in folder: {}'.format(result_path))
        return redirect(url_for('filtering_index'))

    @app.route("/analyze", methods=['POST'])
    def analyze():
        args = request.form
        title = args['title']
        model_id = args['model_id']
        cut_size = int(args['cut_size'])
        overlap = float(args['overlap'])
        custom_params = args['custom_params'].split(',')

        if args['image_loader'] == 'input':
            image = request.files["image"]
            image = cv2.imdecode(numpy.fromstring(image.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)

        if args['image_loader'] == 'mapbox':
            mapbox_id = args["mapbox"]
            mapbox_entry = get_mapbox_by_id(mapbox_id)
            image = mapbox_entry['path']
            image = load_image_by_name(image)

        result_path = analyze_image(image, model_id, title, overlap, cut_size, custom_params)
        flash('Input was successfully analyzed and results are stored in folder: {}'.format(result_path))

        if args.get('download_result') is not None:
            zip_file = shutil.make_archive(os.path.join(os.getcwd(), Path(result_path).stem), 'zip', result_path)
            path_stripped_from_root = os.sep.join(zip_file.split(os.sep)[1:])
            return send_file(path_stripped_from_root, as_attachment=True)
        return redirect(url_for('index'))

    @app.route("/models")
    def models_index():
        return render_template('models/index.html', models=models_repository.get_all())

    @app.route("/models/new", methods=('GET', 'POST'))
    def model_create():
        errors = {}

        if request.method == 'GET':
            return render_template('models/new.html', errors=errors, data={})

        # <errors = validate(request.form)

        if len(errors) > 0:
            return render_template('models/new.html', errors=errors, data=request.form)

        title = request.form.get('title')
        description = request.form.get('description')
        file = request.files["file"]

        store_model(title, description, file)
        return redirect(url_for('models_index'))

    @app.route("/model/<int:id>/delete", methods=['POST'])
    def model_delete(id):
        models_repository.delete(id)
        return redirect(url_for('models_index'))

    @app.route("/mapbox")
    def mapbox_index():
        return render_template('mapbox/index.html', mapboxes=get_all_mapboxes())

    @app.route("/mapbox/<int:id>")
    def mapbox_show(id):
        data = get_mapbox_by_id(id)

        return app.response_class(
            response=json.dumps(data),
            status=200,
            mimetype='application/json'
        )

    @app.route("/mapbox/new", methods=('GET', 'POST'))
    def mapbox_create():
        errors = {}

        if request.method == 'GET':
            return render_template('mapbox/new.html', errors=errors, data={})

        errors = validate(request.form)

        if len(errors) > 0:
            return render_template('mapbox/new.html', errors=errors, data=request.form)

        title = request.form.get('title')
        description = request.form.get('description')
        coordinates = request.form.get('coordinates').split(';')
        coordinates = [float(coordinate) for coordinate in coordinates]

        store_composite(title, description, coordinates)
        return redirect(url_for('mapbox_index'))

    @app.route("/mapbox/<int:id>/delete", methods=['POST'])
    def mapbox_delete(id):
        delete_composite(id)
        return redirect(url_for('mapbox_index'))

    return app


if __name__ == "__main__":
    app = create_app()
    app.run()
