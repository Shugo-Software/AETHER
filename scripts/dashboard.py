#!/usr/bin/env python3
"""AETHER dashboard part 2/2: HTML render + HTTP server.

Stdlib-only Python + inline vanilla JS (no external libs). The page polls
/api/snapshot every 5s and redraws the live panels (stat cards, loss curve,
model health, GPUs, generations). A 60s meta-refresh is a no-JS fallback.

Control plane: POST /api/train/start and /api/train/stop drive student
training; logic lives in train_control.py. The bind address is chosen with
--host (default 127.0.0.1). Binding elsewhere (e.g. 0.0.0.0) exposes the
control plane to that network, so scope the firewall accordingly.
"""
import html
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_CTL_LOCK = threading.Lock()  # serialize train start/stop (no double-launch race)
BIND_HOST = "127.0.0.1"  # actual bind host; set in main(), echoed on the page

PAGE_TOP = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="60">
<title>AETHER training dashboard</title>
<style>
body{font-family:system-ui,Segoe UI,Roboto,sans-serif;background:#111418;
color:#e6e9ee;margin:0 auto;max-width:1000px;padding:24px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px;color:#9fb4d8}
h3{font-size:14px;margin:18px 0 6px;color:#b9c6dd}
.small{color:#8b93a1;font-size:12px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid #2a2f38}
.done{color:#58c184;font-weight:700}
.active{color:#f0b429;font-weight:700}.todo{color:#8b93a1}
.bar{height:10px;background:#2a2f38;border-radius:5px;overflow:hidden;min-width:120px}
.bar>div{height:100%;background:#4aa3ff}
pre{background:#0b0d10;border:1px solid #2a2f38;border-radius:6px;
padding:10px;overflow:auto;font-size:12px;max-height:300px;white-space:pre-wrap}
.card{background:#181c22;border:1px solid #2a2f38;border-radius:8px;
padding:12px 14px;margin:8px 0}
.ok{color:#58c184}.bad{color:#e5625e}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin:10px 0}
.stat{background:#181c22;border:1px solid #2a2f38;border-radius:8px;
padding:10px 16px;min-width:140px}
.stat .k{color:#8b93a1;font-size:12px}
.stat .v{font-size:22px;font-weight:700;margin-top:2px}
.ranges{display:flex;gap:6px;margin:6px 0}
.ranges button{background:#2a2f38;color:#e6e9ee;border:1px solid #2a2f38;
border-radius:6px;padding:3px 12px;cursor:pointer;font-size:12px}
.ranges button.on{background:#4aa3ff;color:#0b0d10;border-color:#4aa3ff}
.chartbox{background:#0b0d10;border:1px solid #2a2f38;border-radius:8px;padding:6px}
.chartbox svg{width:100%;height:auto;display:block}
.gen{display:flex;gap:12px;flex-wrap:wrap}
.gen select{background:#181c22;color:#e6e9ee;border:1px solid #2a2f38;
border-radius:6px;padding:6px;max-width:340px}
.gen .out{flex:1;min-width:280px}
.ctl{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end}
.ctl label{display:flex;flex-direction:column;font-size:12px;color:#8b93a1;gap:3px}
.ctl select,.ctl input{background:#0b0d10;color:#e6e9ee;border:1px solid #2a2f38;
border-radius:6px;padding:6px 8px;font-size:13px}
.ctl input{width:90px}
button.act{cursor:pointer;border:none;border-radius:6px;padding:8px 16px;
font-size:14px;font-weight:700}
button.start{background:#58c184;color:#0b0d10}
button.stop{background:#e5625e;color:#fff}
button.act:disabled{opacity:.4;cursor:not-allowed}
#ctl-status{font-size:13px;margin-top:8px;color:#b9c6dd}
</style></head><body>
<h1>AETHER training dashboard</h1>
"""


def _step_total(m):
    """'current / total' step label; '?' until total is known."""
    steps = m.get("steps") or []
    cur = steps[-1] if steps else 0
    total = m.get("total")
    return "%s / %s" % (cur, total if total else "?")


def _training_header(snap):
    """Stat cards + status + progress — rendered at the very top of the page."""
    t = snap["training"]
    tr = t.get("detail", {}) if isinstance(t.get("detail"), dict) else {}
    status = tr.get("status", "not started")
    tag = tr.get("tag", "-")
    dataset = tr.get("dataset", "-")
    out = tr.get("out", "-")
    loss = tr.get("train_loss")
    samples = tr.get("samples", "-")
    m = t.get("metrics", {})
    step_time = (str(m.get("step_time_s")) + "s") if m.get("step_time_s") is not None else "-"
    eta = _fmt_secs(m.get("eta_sec"))

    h = ["<h2>Training</h2>"]
    h.append('<div class="stats">'
             '<div class="stat"><div class="k">step</div>'
             '<div class="v" id="stat-step">%s</div></div>'
             '<div class="stat"><div class="k">last loss</div>'
             '<div class="v" id="stat-loss">%s</div></div>'
             '<div class="stat"><div class="k">step time</div>'
             '<div class="v" id="stat-time">%s</div></div>'
             '<div class="stat"><div class="k">time remaining</div>'
             '<div class="v" id="stat-eta">%s</div></div></div>'
             % (_step_total(m), _loss_now(m), step_time, eta))
    h.append('<div class="card">status <b>%s</b> &middot; tag %s &middot; '
             'dataset %s &middot; output %s</div>'
             % (html.escape(str(status)), html.escape(str(tag)),
                html.escape(str(dataset)), html.escape(str(out))))
    if status == "done" and loss is not None:
        try:
            h.append('<div class="card ok">final loss %.4f</div>' % float(loss))
        except (ValueError, TypeError):
            pass
    elif status == "running":
        h.append('<div class="card active">training &middot; log: %s</div>'
                 % html.escape(str(t.get("log_file", "-"))))
    if samples != "-":
        h.append('<div class="small">samples <b>%s</b></div>' % html.escape(str(samples)))
    steps = m.get("steps") or []
    total = m.get("total")
    if steps and total:
        try:
            pct = 100.0 * float(steps[-1]) / float(total)
            h.append('<div class="bar"><div style="width:%.1f%%"></div></div>'
                     '<div class="small">%.1f%% complete</div>' % (pct, pct))
        except (ValueError, ZeroDivisionError):
            pass
    return h


def render(snap):
    t = snap["training"]
    tr = t.get("detail", {}) if isinstance(t.get("detail"), dict) else {}
    status = tr.get("status", "not started")
    tag = tr.get("tag", "-")
    dataset = tr.get("dataset", "-")
    out = tr.get("out", "-")
    loss = tr.get("train_loss")
    samples = tr.get("samples", "-")
    m = t.get("metrics", {})
    cfg = t.get("config")

    h = [PAGE_TOP]
    h.append('<div class="small">live 5s poll (no-JS fallback 60s), '
             "bind %s, " % html.escape(BIND_HOST) +
             "snapshot " + html.escape(snap["updated_utc"]) + "</div>")

    # Training (very top): stat cards + status + progress
    h.extend(_training_header(snap))

    # Training controls (start/stop/select dataset)
    ctrl = snap.get("control", {})
    h.append("<h2>Training controls</h2>")
    h.append('<div class="card"><div class="ctl">')
    h.append('<label>dataset<select id="ctl-dataset">')
    for d in ctrl.get("datasets", []):
        h.append('<option value="%s">%s (%s samples)</option>'
                 % (html.escape(d["file"]), html.escape(d["file"]), d["samples"]))
    h.append("</select></label>")
    h.append('<label>resume from<select id="ctl-resume">'
             '<option value="">(fresh LoRA)</option>')
    for a in ctrl.get("adapters", []):
        h.append('<option value="%s">%s (%s MB)</option>'
                 % (html.escape(a["dir"]), html.escape(a["dir"]), a["mb"]))
    h.append("</select></label>")
    h.append('<label>tag<input id="ctl-tag" value="jadis-1.0" maxlength="40"></label>')
    h.append('<label>adapter dir<input id="ctl-out" placeholder="(optional) jadis-1.0" maxlength="40"></label>')
    h.append('<label>epochs<input id="ctl-epochs" type="number" step="0.5" value="3"></label>')
    h.append('<label>max-len<input id="ctl-maxlen" type="number" value="1024"></label>')
    h.append('<label>lr<input id="ctl-lr" value="2e-4"></label>')
    h.append('<button class="act start" id="ctl-start">Start training</button>')
    h.append('<button class="act stop" id="ctl-stop" disabled>Stop training</button>')
    h.append("</div>")
    h.append('<div id="ctl-status">idle — pick a dataset and press Start</div>')
    h.append("</div>")

    # Export / push (merge LoRA -> standalone model -> optional GGUF/HF)
    h.append("<h2>Export / push</h2>")
    h.append('<div class="card"><div class="ctl">')
    h.append('<label>adapter<select id="exp-adapter">')
    for a in ctrl.get("adapters", []):
        h.append('<option value="%s">%s (%s MB)</option>'
                 % (html.escape(a["dir"]), html.escape(a["dir"]), a["mb"]))
    h.append("</select></label>")
    import export_model as _em  # single source of truth for the quant list
    h.append('<label>gguf<select id="exp-gguf">'
             '<option value="">no</option>')
    for q in sorted(_em.QUANT_TYPES):
        label = q + (" (needs imatrix)" if q in _em.IMATRIX_QUANTS else "")
        h.append('<option value="%s">%s</option>' % (q, label))
    h.append('</select></label>')
    h.append('<label>hub<select id="exp-hub">'
             '<option value="">no push</option>'
             '<option value="private">push private</option>'
             '<option value="public">push public</option></select></label>')
    h.append('<label>repo<input id="exp-repo" placeholder="(auto owner/name)"></label>')
    h.append('<button class="act start" id="exp-start">Export model</button>')
    h.append('<button class="act stop" id="exp-stop" disabled>Stop export</button>')
    h.append("</div>")
    h.append('<div id="exp-status">idle — merge writes exports/&lt;adapter&gt;/ '
             '(fp16 safetensors, loads anywhere; push needs a one-time '
             'huggingface-cli login)</div>')
    h.append('<pre class="out" id="exp-log" style="display:none"></pre>')
    h.append("</div>")

    # Phases
    h.append("<h2>Phases</h2><table><tr><th>#</th><th>Goal</th>"
             "<th>Status</th><th>Doc</th></tr>")
    for p in snap["phases"]:
        doc = p["doc"] if p["doc"] else "-"
        h.append("<tr><td>%s</td><td>%s</td>" % (p["n"], html.escape(p["name"])))
        h.append('<td class="%s">%s</td><td>%s</td></tr>'
                 % (p["status"], p["status"], doc))
    h.append("</table>")

    # Live GPUs (Machine)
    h.append("<h2>Machine (GPUs, live)</h2>")
    g = snap["gpus"]
    if g and "error" not in g[0]:
        h.append('<table><tr><th>#</th><th>GPU</th><th>Util</th>'
                 "<th>VRAM</th><th></th></tr>")
        h.append('<tbody id="gpu-body">')
        for gpu in g:
            h.append('<tr data-idx="%s"><td>%s</td><td>%s</td>'
                     '<td class="u">%.0f%%</td><td class="v">%.0f/%.0f MiB</td>' % (
                         gpu["index"], gpu["index"], html.escape(gpu["name"]),
                         gpu["util"], gpu["used"], gpu["total"]))
            h.append('<td><div class="bar"><div class="p" style="width:%.1f%%">'
                     "</div></div></td></tr>" % gpu["pct"])
        h.append("</tbody></table>")
    else:
        h.append('<div class="card bad">nvidia-smi unavailable</div>')

    # Teacher endpoint
    h.append("<h2>Teacher endpoint (:8080)</h2>")
    t2 = snap["teacher"]
    if t2.get("up"):
        h.append('<div class="card ok">UP health %s, serving %s</div>' % (
            html.escape(t2["health"]),
            html.escape(", ".join(t2["models"]))))
    else:
        h.append('<div class="card bad">DOWN %s</div>'
                 % html.escape(str(t2.get("error"))))

    # Datasets
    h.append("<h2>Datasets</h2>")
    ds = snap["datasets"]
    if ds:
        h.append("<table><tr><th>File</th><th>Tag</th>"
                 "<th>Samples</th><th>Size</th></tr>")
        for d in ds:
            h.append("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s KB</td></tr>"
                     % (html.escape(d["file"]), html.escape(d["tag"]),
                        d["samples"], d["kb"]))
        h.append("</table>")
    else:
        h.append('<div class="card">no datasets yet</div>')

    # Training charts (stat cards + status live at the top of the page)
    h.append("<h2>Training charts</h2>")

    # Run configuration
    h.append("<h3>Run configuration</h3>")
    if cfg:
        h.append('<div class="card"><table>')
        for k in ("student", "dataset", "epochs", "lr", "max_len", "max_steps",
                  "batch", "grad_accum", "lora_r", "lora_alpha", "optim",
                  "precision"):
            if k in cfg:
                h.append("<tr><td>%s</td><td><b>%s</b></td></tr>"
                         % (html.escape(k), html.escape(str(cfg[k]))))
        h.append("</table></div>")
    else:
        h.append('<div class="card small">(no run config recorded yet)</div>')

    # Loss curve (range filter)
    h.append("<h3>Loss curve</h3>")
    h.append('<div class="ranges" id="ranges">'
             '<button data-n="10">Last 10</button>'
             '<button data-n="25">Last 25</button>'
             '<button data-n="50">Last 50</button>'
             '<button data-n="0" class="on">All</button></div>')
    h.append('<div class="chartbox"><svg id="loss-chart"></svg></div>')

    # Model health: grad norm + learning rate
    h.append("<h3>Model health</h3>")
    h.append('<div class="small">grad_norm = how far each update moves the policy</div>')
    h.append('<div class="chartbox"><svg id="grad-chart"></svg></div>')
    h.append('<div class="small">learning rate</div>')
    h.append('<div class="chartbox"><svg id="lr-chart"></svg></div>')

    # Live generations
    h.append("<h3>Live generations</h3>")
    h.append('<div class="gen"><select id="gen-select"></select>'
             '<pre class="out" id="gen-out">(no samples yet - run '
             "scripts/sample_student.py after training)</pre></div>")

    # Live log tail
    h.append("<h3>Live log tail</h3>")
    log = t.get("log_tail", [])
    if log:
        h.append("<pre>" + "\n".join(html.escape(str(x)) for x in log[-20:]) +
                 "</pre>")
    else:
        h.append('<div class="card">(no log yet)</div>')

    # Embed initial data for the client-side poller (no-JS fallback shows server HTML).
    init = {"metrics": m, "samples": t.get("samples", []), "gpus": snap["gpus"],
            "control": ctrl, "export": snap.get("export", {})}
    h.append("<script>window.INIT=" +
             json.dumps(init, ensure_ascii=False).replace("</", "<\\/") + ";</script>")
    h.append(JS_APP)
    h.append("</body></html>")
    return "\n".join(h)


def _loss_now(m):
    losses = m.get("losses") or []
    return "%.4f" % losses[-1] if losses else "-"


def _fmt_secs(sec):
    if sec is None:
        return "-"
    sec = int(sec)
    hh, rem = divmod(sec, 3600)
    mm, ss = divmod(rem, 60)
    if hh:
        return "%dh %dm" % (hh, mm)
    if mm:
        return "%dm %ds" % (mm, ss)
    return "%ds" % ss


JS_APP = """
<script>
var NS='http://www.w3.org/2000/svg';
var range=0, last={steps:[],losses:[],grad_norms:[],lrs:[]}, genSel=-1;
function el(t){return document.createElementNS(NS,t);}
function drawChart(svg,xs,ys,label){
  if(!svg)return; while(svg.firstChild)svg.removeChild(svg.firstChild);
  if(!xs||!xs.length){var t0=el('text');t0.setAttribute('x',8);t0.setAttribute('y',20);
    t0.setAttribute('fill','#8b93a1');t0.setAttribute('font-size','12');
    t0.textContent='(no data yet)';svg.appendChild(t0);return;}
  var W=960,H=170,P=30;
  svg.setAttribute('viewBox','0 0 '+W+' '+H);
  var ymin=Math.min.apply(null,ys),ymax=Math.max.apply(null,ys);
  if(ymax-ymin<1e-9){ymax=ymin+1;}
  var x0=xs[0],x1=xs[xs.length-1]; if(x1===x0)x1=x0+1;
  function X(v){return P+(v-x0)/(x1-x0)*(W-2*P);}
  function Y(v){return H-P-(v-ymin)/(ymax-ymin)*(H-2*P);}
  var pts=[]; for(var i=0;i<xs.length;i++)pts.push(X(xs[i]).toFixed(1)+','+Y(ys[i]).toFixed(1));
  var pl=el('polyline'); pl.setAttribute('points',pts.join(' '));
  pl.setAttribute('fill','none'); pl.setAttribute('stroke','#4aa3ff'); pl.setAttribute('stroke-width','2');
  svg.appendChild(pl);
  var c=el('circle'); c.setAttribute('cx',X(xs[xs.length-1])); c.setAttribute('cy',Y(ys[ys.length-1]));
  c.setAttribute('r','3.5'); c.setAttribute('fill','#f0b429'); svg.appendChild(c);
  var t1=el('text'); t1.setAttribute('x',P); t1.setAttribute('y',H-8); t1.setAttribute('fill','#8b93a1');
  t1.setAttribute('font-size','11'); t1.textContent=label+'  step '+x0+' -> '+x1; svg.appendChild(t1);
  var t2=el('text'); t2.setAttribute('x',P); t2.setAttribute('y',16); t2.setAttribute('fill','#8b93a1');
  t2.setAttribute('font-size','11');
  t2.textContent=label+'  max '+ymax.toFixed(4)+'   min '+ymin.toFixed(4); svg.appendChild(t2);
}
function drawLoss(){
  var n=last.steps.length, s2=last.steps, l2=last.losses;
  if(range>0&&n>range){s2=last.steps.slice(n-range);l2=last.losses.slice(n-range);}
  drawChart(document.getElementById('loss-chart'),s2,l2,'loss');
}
function fmtSecs(sec){ if(sec==null)return '-'; sec=parseInt(sec);
  var h=Math.floor(sec/3600),m=Math.floor((sec%3600)/60),s=sec%60;
  if(h)return h+'h '+m+'m'; if(m)return m+'m '+s+'s'; return s+'s'; }
function updateGpus(gpus){
  var body=document.getElementById('gpu-body'); if(!body||!gpus)return;
  for(var i=0;i<gpus.length;i++){var gpu=gpus[i];
    var row=body.querySelector('tr[data-idx="'+gpu.index+'"]'); if(!row)continue;
    var u=row.querySelector('.u'),v=row.querySelector('.v'),p=row.querySelector('.p');
    if(u)u.textContent=Math.round(gpu.util)+'%';
    if(v)v.textContent=Math.round(gpu.used)+'/'+Math.round(gpu.total)+' MiB';
    if(p)p.style.width=gpu.pct+'%';}
}
function buildGens(samples){
  var sel=document.getElementById('gen-select'),out=document.getElementById('gen-out');
  if(!sel)return; if(!samples||!samples.length){sel.style.display='none';return;}
  var keep=genSel; sel.innerHTML=''; sel.style.display='';
  samples.forEach(function(s,i){
    var o=document.createElement('option'); o.value=i;
    o.textContent=(s.recipe?s.recipe+': ':'')+(s.prompt||'').slice(0,60);
    sel.appendChild(o);});
  if(keep>=0&&keep<samples.length)sel.value=keep;
  showGen();
}
function showGen(){
  var sel=document.getElementById('gen-select'),out=document.getElementById('gen-out');
  var samples=window.__samples||[]; if(!sel||!out)return;
  var i=parseInt(sel.value); genSel=i; var s=samples[i]; if(!s)return;
  out.textContent=(s.prompt||'')+'\\n\\n--- '+s.recipe+' ('+s.model+') ---\\n\\n'+(s.completion||'');
}
function esc(s){var d=document.createElement('div');d.appendChild(document.createTextNode(s));return d.innerHTML;}
function apply(s){
  var m=(s.training&&s.training.metrics)||{}; last=m;
  var st=document.getElementById('stat-step');
  if(st){var cur=(m.steps&&m.steps.length)?m.steps[m.steps.length-1]:0;
    st.textContent=cur+' / '+(m.total?m.total:'?');}
  var sl=document.getElementById('stat-loss'); if(sl)sl.textContent=m.losses&&m.losses.length?m.losses[m.losses.length-1].toFixed(4):'-';
  var stt=document.getElementById('stat-time'); if(stt)stt.textContent=(m.step_time_s!=null)?m.step_time_s+'s':'-';
  var se=document.getElementById('stat-eta'); if(se)se.textContent=fmtSecs(m.eta_sec);
  drawLoss();
  drawChart(document.getElementById('grad-chart'),m.steps,m.grad_norms,'grad_norm');
  drawChart(document.getElementById('lr-chart'),m.steps,m.lrs,'lr');
  updateGpus(s.gpus);
  if(s.training&&s.training.samples){window.__samples=s.training.samples;buildGens(s.training.samples);}
  if(s.control)applyControl(s.control);
  if(s.export)updateExport(s.export);
}
function applyControl(c){
  var st=document.getElementById('ctl-start'),sp=document.getElementById('ctl-stop');
  var msg=document.getElementById('ctl-status'); if(!st||!sp||!msg)return;
  if(c.active){
    st.disabled=true; sp.disabled=false;
    msg.innerHTML='<span class="active">training</span> tag='+(c.active.tag||'?')+
      ' pid='+c.active.pid+' dataset='+(c.active.dataset||'?')+' since '+(c.active.started_utc||'');
  }else{
    st.disabled=false; sp.disabled=true;
    msg.textContent=c.can_start?'idle — pick a dataset and press Start':'busy…';
  }
}
function ctlMsg(t,ok){var m=document.getElementById('ctl-status');
  m.innerHTML=ok?('<span class="ok">'+t+'</span>'):('<span class="bad">'+t+'</span>');}
function startTrain(){
  var body={dataset:document.getElementById('ctl-dataset').value,
    resume:document.getElementById('ctl-resume').value,
    tag:document.getElementById('ctl-tag').value.trim(),
    out:document.getElementById('ctl-out').value.trim(),
    epochs:parseFloat(document.getElementById('ctl-epochs').value),
    max_len:parseInt(document.getElementById('ctl-maxlen').value),
    lr:document.getElementById('ctl-lr').value.trim()};
  ctlMsg('starting…',true);
  fetch('/api/train/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)}).then(function(r){return r.json();}).then(function(o){
    if(o.ok){ctlMsg('started pid='+o.pid+' tag='+o.tag+' — watch the panels below',true);}
    else{ctlMsg('error: '+(o.error||'failed'),false);} poll();
  }).catch(function(e){ctlMsg('request failed: '+e,false);});
}
function stopTrain(){
  ctlMsg('stopping…',true);
  fetch('/api/train/stop',{method:'POST'}).then(function(r){return r.json();}).then(function(o){
    if(o.ok){ctlMsg('stopped'+(o.tag?(' '+o.tag):'')+' ('+(o.note||'')+')',true);}
    else{ctlMsg('error: '+(o.error||'failed'),false);} poll();
  }).catch(function(e){ctlMsg('request failed: '+e,false);});
}
function startExport(){
  var hub=document.getElementById('exp-hub').value;
  var body={adapter:document.getElementById('exp-adapter').value,
    gguf:document.getElementById('exp-gguf').value,
    hub:hub==='private'||hub==='public',
    public:hub==='public',
    repo:document.getElementById('exp-repo').value.trim()};
  var el=document.getElementById('exp-status');
  el.innerHTML='<span class="ok">starting export…</span>';
  fetch('/api/export/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)}).then(function(r){return r.json();}).then(function(o){
    if(!o.ok){el.innerHTML='<span class="bad">error: '+(o.error||'failed')+'</span>';}
    poll();
  }).catch(function(e){el.innerHTML='<span class="bad">request failed: '+e+'</span>';});
}
function stopExport(){
  var el=document.getElementById('exp-status');
  el.innerHTML='<span class="ok">stopping…</span>';
  fetch('/api/export/stop',{method:'POST'}).then(function(r){return r.json();}).then(function(o){
    if(!o.ok){el.innerHTML='<span class="bad">error: '+(o.error||'failed')+'</span>';}
    poll();
  }).catch(function(e){el.innerHTML='<span class="bad">request failed: '+e+'</span>';});
}
function updateExport(x){
  var sb=document.getElementById('exp-start'), tb=document.getElementById('exp-stop');
  if(sb)sb.disabled=!!x.active; if(tb)tb.disabled=!x.active;
  var el=document.getElementById('exp-status'); if(!el)return;
  var s=x.status;
  if(x.active&&s){el.innerHTML='<span class="ok">exporting '+
    esc(String(s.adapter||''))+' — '+esc(String(s.status||''))+
    (s.stage?(' ('+esc(String(s.stage))+')'):'')+'</span>';}
  else if(s&&s.status==='failed'){el.innerHTML='<span class="bad">last export of '+
    esc(String(s.adapter||''))+' failed: '+esc(String(s.error||''))+'</span>';}
  else if(s&&s.status==='done'){el.innerHTML='<span class="ok">last export done: '+
    esc(String(s.out||''))+(s.hub_url?(' → '+esc(s.hub_url)):'')+'</span>';}
  var lg=document.getElementById('exp-log');
  if(lg){if(x.log_tail&&x.log_tail.length){lg.style.display='block';
    lg.textContent=x.log_tail.join('\n');}else{lg.style.display='none';}}
}
function poll(){
  fetch('/api/snapshot').then(function(r){return r.json();}).then(apply).catch(function(){});
}
document.addEventListener('DOMContentLoaded',function(){
  if(window.INIT){window.__samples=window.INIT.samples||[];
    apply({training:{metrics:window.INIT.metrics,samples:window.INIT.samples},
           gpus:window.INIT.gpus,control:window.INIT.control,export:window.INIT.export});}
  var gs=document.getElementById('gen-select'); if(gs)gs.addEventListener('change',showGen);
  var rg=document.getElementById('ranges'); if(rg)rg.addEventListener('click',function(e){
    if(e.target.tagName!=='BUTTON')return; range=parseInt(e.target.getAttribute('data-n'));
    var bs=rg.querySelectorAll('button'); for(var i=0;i<bs.length;i++)bs[i].classList.remove('on');
    e.target.classList.add('on'); drawLoss();});
  var sb=document.getElementById('ctl-start'); if(sb)sb.addEventListener('click',startTrain);
  var tb=document.getElementById('ctl-stop'); if(tb)tb.addEventListener('click',stopTrain);
  var eb=document.getElementById('exp-start'); if(eb)eb.addEventListener('click',startExport);
  var xs=document.getElementById('exp-stop'); if(xs)xs.addEventListener('click',stopExport);
  poll(); setInterval(poll,5000);
});
</script>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        from dashdata import snapshot as _snap
        if self.path == "/api/snapshot":
            self._json(200, _snap())
        else:
            body = render(_snap()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_POST(self):
        import train_control as tc
        if self.path == "/api/train/start":
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                cfg = json.loads(raw or b"{}")
            except ValueError:
                self._json(400, {"ok": False, "error": "invalid JSON body"})
                return
            # Serialize start/stop so two clicks can't race a double-launch.
            with _CTL_LOCK:
                ok, payload = tc.start_training(cfg)
            self._json(200 if ok else 409, dict(payload, ok=ok))
        elif self.path == "/api/train/stop":
            with _CTL_LOCK:
                ok, payload = tc.stop_training()
            self._json(200 if ok else 409, dict(payload, ok=ok))
        elif self.path == "/api/export/start":
            import export_control as ec
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                cfg = json.loads(raw or b"{}")
            except ValueError:
                self._json(400, {"ok": False, "error": "invalid JSON body"})
                return
            with _CTL_LOCK:
                ok, payload = ec.start_export(cfg)
            self._json(200 if ok else 409, dict(payload, ok=ok))
        elif self.path == "/api/export/stop":
            import export_control as ec
            with _CTL_LOCK:
                ok, payload = ec.stop_export()
            self._json(200 if ok else 409, dict(payload, ok=ok))
        else:
            self._json(404, {"ok": False, "error": "unknown endpoint"})


def main():
    import argparse
    global BIND_HOST
    ap = argparse.ArgumentParser(description="AETHER dashboard")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default 127.0.0.1 = loopback only; "
                         "0.0.0.0 exposes to all networks - scope the firewall!)")
    args = ap.parse_args()
    BIND_HOST = args.host
    try:
        srv = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        print("cannot bind %s:%d (%s); is the interface up? "
              "(a Tailscale IP bind needs Tailscale running)"
              % (args.host, args.port, exc), flush=True)
        return 1
    if args.host in ("0.0.0.0", "::"):
        print("dashboard on all interfaces - http://127.0.0.1:%d/ (this machine)"
              % args.port, flush=True)
    else:
        print("dashboard at http://%s:%d/" % (args.host, args.port), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

