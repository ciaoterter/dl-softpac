from flask import Blueprint, render_template, request, Response
import cv2
import maskfind

views = Blueprint(__name__,"views")

model, device = maskfind.main()

@views.route('/', methods=['GET', 'POST'])
def home():
    value = None

    if request.method == 'POST':
        submit_button = request.form.get('run', None)
        reset_button = request.form.get('reset', None)

        if submit_button:
            # Do your stuff
            value = 'Some result if submit_button_1 was pressed'
        elif reset_button:
            # Do your stuff
            value = None

    return render_template(
        'index.html',
        value=value,
    )

@views.route('/image')
def image():
    #img = cv2.imread('test.jpg')
    img = maskfind.segment_instance('test.jpg', model, device, confidence=0.6)
    data = cv2.imencode('.jpg', img)[1].tobytes()
    return Response(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n', mimetype='multipart/x-mixed-replace; boundary=frame')
