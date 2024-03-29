import os
import re
import csv
import json, jsonify
import requests
import numpy as np

from glob import glob
from time import sleep
from flask import Flask, request, jsonify, render_template, url_for, redirect
from flask import  send_from_directory, send_file
from flask import session

# for reval
import torch
from ast import literal_eval
import tifffile

from cppn import CPPN, run_cppn


APPNAME = 'NealeFlaskCPPNSite'
app = Flask(__name__, static_url_path='', static_folder='static')
app.config.update(APPNAME=APPNAME)
app.secret_key = str(np.random.randint(int(1e9)))
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


def init_cppn(uid, rand=False):
    if rand:
        session['cppn_config']['seed'] = np.random.randint(123456789)
    return CPPN(**session['cppn_config'])
 

@app.route('/home')
def home():
    try:
        clear_images(session['uid'], 'jpg')
        clear_images(session['uid'], 'tif')
    except Exception as e:
        print (e)

    return render_template("index.html")


@app.route('/profile')
def profile():
    return render_template("index.html")

@app.route('/genart')
def genart():
    return render_template("genart.html")

@app.route('/gallery')
def gallery():
    return render_template("gallery.html")

@app.route('/hyperevolve')
def hyperevolve():
    return render_template("hyperevolve.html")

@app.route('/digitalflow')
def digitalflow():
    return render_template("digitalflow.html")

@app.route('/haiku')
def gallery_haiku():
    return render_template("haiku.html")

@app.route('/untitled')
def gallery_untitled():
    return render_template("untitled.html")

@app.route('/windows')
def gallery_windows():
    return render_template("windows.html")

@app.route('/windows2')
def gallery_windows2():
    return render_template("windows2.html")

@app.route('/flow1')
def gallery_flow1():
    return render_template("flow-series1.html")

@app.route('/flow2')
def gallery_flow2():
    return render_template("flow-series2.html")

@app.route('/flow3')
def gallery_flow3():
    return render_template("flow-series3.html")

@app.route('/llmashup')
def gallery_llmashup():
    return render_template("llmashup.html")

@app.route('/animations')
def animations():
    return render_template("animations.html")

def get_initial_config():
   return  {'z_dim'    : 4,
            'n_samples': 6,
            'x_dim': 512,
            'y_dim': 512,
            'c_dim': 3,
            'batch_size': 16,
            'z_scale': 4,
            'layer_width': 4,
            'interpolation': 10,
            'exp_name': 'static/assets/cppn_images',
            'seed': None,
            'seed_gen': 1234567890,
            'uid': session['uid']}

@app.route('/cppn')
def cppn_viewer():
    # start CPPN 
    if 'uid' not in session.keys():
        uid = np.random.randint(int(1e9))
        session['uid'] = uid
    if 'cppn_config' not in session.keys():
        print('creating cppn config')
        session['cppn_config'] = get_initial_config()
    session['curr_img_idx'] = 0
    session['landing_img'] = 'images/12.png'
    session['landing_img_sm'] = 'images/12_sm.png'
    session['last_act_order'] = []
    
    slider_vals = {
        'z_slider_init'            : session['cppn_config']['z_dim'],
        'layer_width_slider_init'  : session['cppn_config']['layer_width'],
        'z_scale_slider_init'      : session['cppn_config']['z_scale'],
        'interpolation_range_init' : session['cppn_config']['interpolation'],}
    
    return render_template('cppn.html',
                           data={'landing': session['landing_img'],
                                 'landing_sm': session['landing_img_sm']},
                           slider_vals=slider_vals)

@app.route('/cppn-simple')
def cppn_viewer_simple():
    # start CPPN 
    if 'uid' not in session.keys():
        uid = np.random.randint(int(1e9))
        session['uid'] = uid
    if 'cppn_config' not in session.keys():
        print('creating cppn config')
        session['cppn_config'] = get_initial_config()
    session['curr_img_idx'] = 0
    session['landing_img'] = 'images/12.png'
    session['landing_img_sm'] = 'images/12_sm.png'
    session['last_act_order'] = []

    return render_template('cppn-simple.html',
            data={'landing': session['landing_img'],
                'landing_sm': session['landing_img_sm']})


@app.route('/cppn-dev')
def cppn_viewer_dev():
    # start CPPN 
    if 'uid' not in session.keys():
        uid = np.random.randint(int(1e9))
        session['uid'] = uid
    if 'cppn_config' not in session.keys():
        print('creating cppn config')
        session['cppn_config'] = get_initial_config()
    session['curr_img_idx'] = 0
    session['landing_img'] = 'images/12.png'
    session['landing_img_sm'] = 'images/12_sm.png'
    session['last_act_order'] = []

    return render_template('cppn-simple.html',
            data={'landing': session['landing_img'],
                'landing_sm': session['landing_img_sm']})


def sort_fn_nums(fns):
    """Sorts files and returns the largest, smallest values according
        to the number (index) at the end of the filename

    Args:
        fns (list): file names to be sorted

    Returns:
        f_min (str): smallest filename according to index.
        f_max (str): largest filename according to index.
    """
    fns_idx = lambda f: int(f.split('_')[-1].split('.')[0])
    sorted_fns = sorted(fns, key=fns_idx)
    # print (sorted_fns)
    f_min = sorted_fns[0]
    f_max = sorted_fns[-1]
    return f_min, f_max, sorted_fns


def clear_images(uid, suffix='jpg'):
    assert suffix in ['jpg', 'tif']
    sample_dir = 'static/assets/cppn_images/{}/temp'.format(uid)
    print (f'clearing {suffix} files from {sample_dir}')
    if suffix == 'jpg':
        for image in glob('{}/*.jpg'.format(sample_dir)):
            os.remove(image)
    elif suffix == 'tif':
        for image in glob('{}/*.tif'.format(sample_dir)):
            os.remove(image)


def run_image_batch(cppn, uid, autosave, z=None):
    sample, fn_suff = run_cppn(cppn, uid, autosave=autosave, z=z)
    return sample, fn_suff


@app.route('/generate-image', methods=['GET', 'POST'])
def generate_image(clear=False, random_gen=False):
    """
    Generates and renders a single image (no ajax jet so page is refreshed)
    """
    print (clear, random_gen)
    print (request.args.get('random_gen'), request.args.get('clear'))
    autosave = True
    if request.method == 'GET':
        return redirect("/cppn")

    elif request.method == 'POST':
        if 'cppn_config' not in session.keys():
            return redirect("/cppn")
        
        uid = session['uid']
        print ('gen image', session['cppn_config'])
        print ('request.form:', request.form)
        sample_dir = 'static/assets/cppn_images/{}/temp/'.format(uid)
        # More button, generate samples by jittering a single z sample
        # chosen sample is the index
        override = False
        if request.form.get('keepseed') == 'true':
            idx  = session['curr_img_idx']
            imgs = glob('{}/*.tif'.format(sample_dir))
            jpgs = glob('{}/*.jpg'.format(sample_dir))
            fn_exist = [fn for fn in jpgs if f'_{idx}.' in fn]
            if len(fn_exist) == 0:  # if we pressed more before generate, run generate
                session['curr_img_idx'] = 1
                session.modified = True
                cppn = init_cppn(uid=uid, rand=True)
                override = True
            else:    
                img_prefix = imgs[0].split('.tif')[0]
                print (img_prefix)
                img_path = img_prefix[:-1]+str(idx)+'.tif'
                with tifffile.TiffFile(img_path) as tif:
                    metadata = tif.shaped_metadata[0]
                session['cppn_config']['seed'] = int(metadata['seed'])
                noise = literal_eval(metadata['z_sample'])
                session.modified = True
                cppn = init_cppn(uid=uid, rand=False)

        else:  # normal generate following ranges
            session['curr_img_idx'] = 1
            session.modified = True
            cppn = init_cppn(uid=uid, rand=True)
        if cppn.generator is None:
            if request.form.get('random-gen') == 'true' or random_gen:
                cppn.init_generator(random_generator=True)
            else:
                cppn.init_generator(random_generator=False)
            os.makedirs(sample_dir, exist_ok=True)
        if request.form.get('clear') == 'true' or clear:
            clear_images(uid, 'jpg') 
            clear_images(uid, 'tif') 
            assert len(glob('{}/*.jpg'.format(sample_dir))) == 0
        if request.form.get('keepseed') == 'true' and not override:
            z = torch.tensor(noise)
            sample, fn_suff = run_image_batch(cppn, uid, autosave, z=z)
        else:
            sample, fn_suff = run_image_batch(cppn, uid, autosave)
        sample, batch_samples = sample
        assert sample is not None, "sample is None"
        images_in_dir = glob('{}/*.jpg'.format(sample_dir))
        images_in_dir = [img for img in images_in_dir if 'lrg' not in img]
        if len(images_in_dir) > 1:
            first_sample, last_sample, samples = sort_fn_nums(images_in_dir)
        else:
            last_sample = images_in_dir[0]
        print (f'new image from _generate_image: {first_sample}')
        first_sample = first_sample.split('static/')[1]
        ret = {'img': first_sample}
        for i in range(1, len(samples)):
            name = samples[i].split('static/')[1]
            ret[f'recent{i}'] = name
        ret['gen_name'] = str(cppn.generator.name)
        
        act_order = [str(a.__repr__()) for a in cppn.generator.acts]
        print ('act order: ', act_order)
        ret['gen_act'] = act_order
        session['last_act_order'] = act_order
        session.modified = True
        return jsonify(ret)
    else:
        return jsonify({'img': 'image_12.png'})


def save_images(x, y, res):
    uid = session['uid']
    idx = session['curr_img_idx']
    sample_dir = f'static/assets/cppn_images/{uid}/temp/'
    tiffs = glob('{}/*.tif'.format(sample_dir))
    print (sample_dir, tiffs)
    img = [img for img in tiffs if f'_{idx}.tif' in img]
    assert len(img) > 0
    path = img[0]
    with tifffile.TiffFile(path) as tif:
        metadata = tif.shaped_metadata[0]

    session['cppn_config']['seed'] = int(metadata['seed'])
    session['cppn_config']['seed_gen'] = int(metadata['seed_gen'])
    session['cppn_config']['z_dim'] = int(metadata['z'])
    session['cppn_config']['z_scale'] = int(float(metadata['scale']))
    session['cppn_config']['layer_width'] = int(metadata['net'])
    session.modified=True
    noise = literal_eval(metadata['z_sample'])
    cppn = init_cppn(uid=session['uid'], rand=False)
    if int(metadata['randgen']) == 1:
        randgen = True
    else:
        randgen = False
    if cppn.generator is None:
        cppn.init_generator(random_generator=randgen)
    sample = cppn.sample_frame(torch.as_tensor(noise),
            x,
            y,
            batch_size=1)
    assert sample is not None, "sample is None"
    sample = sample[0].numpy() * 255.
    name = f'z{cppn.z_dim}_scale{cppn.z_scale}_width{cppn.layer_width}_{res}'
    img_path = f'{sample_dir}{name}_{idx}'
    cppn._write_image(img_path, sample, 'png')
    img_path = img_path+'.png'
    return img_path


@app.route('/download-small', methods=['GET'])
def save_image_small():
    """
    saves an image at a user-specified resolution
    """
    if 'uid' not in session:
        return send_file('static/images/12_sm.png', as_attachment=True)
    sample_dir = 'static/assets/cppn_images/{}/temp/'.format(session['uid'])
    if len(glob('{}/*.jpg'.format(sample_dir))) == 0:
        return send_file('static/images/12_sm.png', as_attachment=True)
    
    x = y = 256
    path = save_images(x, y, 256) 
    return send_file(path, as_attachment=True)


@app.route('/download-lrg', methods=['GET'])
def save_image_large():
    """
    saves an image at a user-specified resolution
    """
    if 'uid' not in session:
        return send_file('static/images/12_lrg.png', as_attachment=True)
    sample_dir = 'static/assets/cppn_images/{}/temp/'.format(session['uid'])
    if len(glob('{}/*.jpg'.format(sample_dir))) == 0:
        return send_file('static/images/12_lrg.png', as_attachment=True)

    x = y = 512
    path = save_images(x, y, 512)
    return send_file(path, as_attachment=True)


@app.route('/download-desktop1', methods=['GET'])
def save_image_desktop1k():
    """
    saves an image at a user-specified resolution
    """
    if 'uid' not in session:
        return send_file('static/images/12_desktop.png', as_attachment=True)
    sample_dir = 'static/assets/cppn_images/{}/temp/'.format(session['uid'])
    if len(glob('{}/*.jpg'.format(sample_dir))) == 0:
        return send_file('static/images/12_desktop.png', as_attachment=True)
    
    x = 1920
    y = 1080
    path = save_images(x, y, '1920-1080')
    return send_file(path, as_attachment=True)

@app.route('/download-desktop2', methods=['GET'])
def save_image_desktop2k():
    """
    saves an image at a user-specified resolution
    """
    if 'uid' not in session:
        return send_file('static/images/12_desktop2k.png', as_attachment=True)
    sample_dir = 'static/assets/cppn_images/{}/temp/'.format(session['uid'])
    if len(glob('{}/*.jpg'.format(sample_dir))) == 0:
        return send_file('static/images/12_desktop2k.png', as_attachment=True)

    x = 2160
    y = 1440
    path = save_images(x, y, '2k')
    return send_file(path, as_attachment=True)


@app.route('/download-desktop4', methods=['GET'])
def save_image_desktop4k():
    """
    saves an image at a user-specified resolution
    """
    if 'uid' not in session:
        return send_file('static/images/12_desktop4k.png', as_attachment=True)
    sample_dir = 'static/assets/cppn_images/{}/temp/'.format(session['uid'])
    if len(glob('{}/*.jpg'.format(sample_dir))) == 0:
        return send_file('static/images/12_desktop4k.png', as_attachment=True)

    x = 3840
    y = 2160
    path = save_images(x, y, '4k')
    return send_file(path, as_attachment=True)


@app.route('/download-square2', methods=['GET'])
def save_image_square2():
    """
    saves an image at a user-specified resolution
    """
    if 'uid' not in session:
        return send_file('static/images/12_desktop4k.png', as_attachment=True)
    sample_dir = 'static/assets/cppn_images/{}/temp/'.format(session['uid'])
    if len(glob('{}/*.jpg'.format(sample_dir))) == 0:
        return send_file('static/images/12_desktop2k.png', as_attachment=True)

    x = 2000
    y = 2000
    path = save_images(x, y, '2square')
    return send_file(path, as_attachment=True)


@app.route('/download-square4', methods=['GET'])
def save_image_square4():
    """
    saves an image at a user-specified resolution
    """
    if 'uid' not in session:
        return send_file('static/images/12_desktop4k.png', as_attachment=True)
    sample_dir = 'static/assets/cppn_images/{}/temp/'.format(session['uid'])
    if len(glob('{}/*.jpg'.format(sample_dir))) == 0:
        return send_file('static/images/12_desktop4k.png', as_attachment=True)

    x = 4000
    y = 4000
    path = save_images(x, y, '4square')
    return send_file(path, as_attachment=True)


@app.route('/regenerate-image', methods=['GET', 'POST'])
def regenerate_image():
    """
    re-Generates and renders a single image (no ajax jet so page is refreshed)
    """
    autosave = True
    if request.method == 'GET':
        return redirect("/cppn")

    elif request.method == 'POST':
        if 'cppn_config' not in session.keys():
            return redirect("/cppn")

        uid = session['uid']
        if request.form.get('set_to') is not None:
            idx = int(request.form.get('set_to')) + 1
        else:
            increment = request.form.get('increment')
            idx = session['curr_img_idx'] + int(increment)
            if idx < 1:
                idx = 6
            if idx > 6:
                idx = 1

        print (f'next idx {idx}')
        sample_dir = f'static/assets/cppn_images/{uid}/temp/'
        
        if len(glob('{}/*.jpg'.format(sample_dir))) == 0:
            print ('no next or prev, redirecting to generate')
            return redirect("/generate-image")
        
        imgs = glob('{}/*.tif'.format(sample_dir))
        jpgs = glob('{}/*.jpg'.format(sample_dir))
        fn_exist = [fn for fn in jpgs if f'lrg{idx}' in fn]
        print (fn_exist)

        if len(fn_exist) > 0:
            print (f'increment to {idx} already exists')
            assert len(fn_exist) < 2
            img_path = fn_exist[0].split('static/')[1]
            session['curr_img_idx'] = idx
            session.modified=True
            return jsonify({'img': f'{img_path}'})

        print ('re-gen image', session['cppn_config'])
        # load image and grab config --> set to session config
        img_prefix = imgs[0].split('.tif')[0]
        img_path = img_prefix[:-1]+str(idx)+'.tif'
        with tifffile.TiffFile(img_path) as tif:
            metadata = tif.shaped_metadata[0]

        session['cppn_config']['seed'] = int(metadata['seed'])
        session['cppn_config']['seed_gen'] = int(metadata['seed_gen'])
        session['cppn_config']['z_dim'] = int(metadata['z'])
        session['cppn_config']['z_scale'] = int(float(metadata['scale']))
        session['cppn_config']['layer_width'] = int(metadata['net'])
        session['curr_img_idx'] = idx
        if int(metadata['randgen']) == 1:
            randgen = True
        else:
            randgen = False

        session.modified = True
        noise = literal_eval(metadata['z_sample'])
        
        cppn = init_cppn(uid=session['uid'], rand=False)
        if cppn.generator is None:
            cppn.init_generator(random_generator=randgen)
        
        sample = cppn.sample_frame(
            torch.as_tensor(noise),
            session['cppn_config']['x_dim'],
            session['cppn_config']['y_dim'],
            batch_size=1)

        assert sample is not None, "sample is None"
        sample = sample[0].numpy() * 255.
        img_path = f'{sample_dir}{np.random.randint(99999)}tmp_lrg{idx}'
        cppn._write_image(img_path, sample, 'jpg')
        img_path = img_path.split('static/')[1]
        ret = {
            'img': f'{img_path}.jpg',
            'img_path_orig': img_prefix
        }
        return jsonify({
            'fn_exist': fn_exist,
            'jpg': jpgs,
            'imgs': imgs,
            'curr idx': session['curr_img_idx'],
            'img_path': img_path,
            'randgen': randgen,
            'img': ret['img'],
            'img_path_orig': ret['img_path_orig'],
            'idx': idx
        })
        return jsonify(ret)
    else:
        return jsonify({'img': 'image_12.png'})


@app.route('/slider-control', methods=['GET', 'POST'])
def slider_control():
    print ('request.form:', request.form)
    if request.method == 'GET':
        print ('get request')
        return jsonify({'nothing': 0})

    elif request.method == 'POST':
        if 'uid' not in session:
            return redirect("/cppn")

        uid  = session['uid']
        ret = {}
        if request.form.get('z_range'):
            new_z = int(request.form.get('z_range'))
            print ('new ', new_z, 'old_z', session['cppn_config']['z_dim'])
            if new_z != session['cppn_config']['z_dim']:
                session['cppn_config']['z_dim'] = new_z
                session.modified = True
            print ('changed to:', session['cppn_config']['z_dim'])
            ret['z'] = new_z

        if request.form.get('net_width_range'):
            new_width = int(request.form.get('net_width_range'))
            if new_width != session['cppn_config']['layer_width']:
                session['cppn_config']['layer_width'] = new_width
                session.modified = True
            ret['net_width'] = new_width

        if request.form.get('z_scale_range'):
            new_z_scale = float(request.form.get('z_scale_range'))
            if new_z_scale != session['cppn_config']['z_scale']:
                session['cppn_config']['z_scale'] = new_z_scale
                session.modified = True
            ret['z_scale'] = new_z_scale
        
        #if request.form.get('interpolation_range'):
        #    new_interpolate = int(request.form.get('interpolation_range'))
        #    if new_interpolate != session['cppn_config']['interpolation']:
        #        session['cppn_config']['interpolation'] = new_interpolate
        #        session.modified = True
        #    ret['interpolation'] = new_interpolate
        else:
            ret['nothing'] = 0
        return jsonify(ret)


@app.route('/vote-image', methods=['GET', 'POST'])
def vote_image():
    print ('request.form:', request.form)
    if request.method == 'GET':
        print ('get request')
        return jsonify({'nothing': 0})

    elif request.method == 'POST':
        if 'uid' not in session:
            return redirect("/cppn")
        uid  = session['uid']
        acts = session['last_act_order']
        print (acts)
        vote = int(request.form.get('vote'))
        print ('what')
        if acts == []:
            print ('quitting')
            return jsonify({})
            #return generate_image(clear=True, random_gen=True)

        if not os.path.isfile('voting_table.csv'):
            with open('voting_table.csv', 'a+') as f:
                # write header
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['Iter','Act0','Act1','Act2','Act3',
                                 'Act4','Act5','Act6','Act7','Act8'])
         
        with open('voting_table.csv', 'r+') as f:
            lines = len(list(csv.reader(f)))
            print ('now {} lines'.format(lines))
            if lines == 0:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['Iter','Act0','Act1','Act2','Act3',
                    'Act4','Act5','Act6','Act7','Act8'])
                lines = 1

        with open('voting_table.csv', 'a+') as f:
            writer = csv.writer(f, delimiter=',')
            row = [str(lines)]
            row.extend(str(vote))
            row.extend(acts[:8])
            writer.writerow(row)

    #return generate_image(clear=True, random_gen=True)
    return jsonify({})


@app.route("/<name>")
def user(name):
    return redirect("/")


@app.route("/")
def root():
    return redirect("/gallery")


if __name__ == '__main__':
    app.run(host='0.0.0.0')
