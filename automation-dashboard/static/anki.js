const $=id=>document.getElementById(id);let dirty=false,subs=[],subsDirty=false;
const SUGGEST=['robotics','MachineLearning','ControlTheory','ROS','math','cpp','compsci','computervision','AskEngineers','LocalLLaMA','reinforcementlearning','embedded','AskAcademia','PhD'];
const norm=v=>v.trim().replace(/^\/?r\//i,'');const validSub=v=>/^[A-Za-z0-9][A-Za-z0-9_]{1,20}$/.test(v);
function renderSubs(){$('tags').innerHTML=subs.map((s,i)=>`<span class="tag"><a href="https://www.reddit.com/r/${safe(s)}/" target="_blank" rel="noopener">r/${safe(s)}</a><button type="button" data-i="${i}" aria-label="r/${safe(s)} 삭제">×</button></span>`).join('');const have=new Set(subs.map(s=>s.toLowerCase()));$('suggest').innerHTML=SUGGEST.filter(s=>!have.has(s.toLowerCase())).map(s=>`<button type="button" class="tag ghost" data-add="${safe(s)}">+ r/${safe(s)}</button>`).join('');$('sub-count').textContent=`${subs.length}/20개 · 저장 후 다음 수집부터 적용`}
function addSub(v){v=norm(v);if(!v)return;if(!validSub(v)){show('서브레딧 이름은 영문·숫자·_ 2~21자입니다.',true);return}if(subs.some(s=>s.toLowerCase()===v.toLowerCase())){show('이미 등록된 서브레딧입니다.');return}if(subs.length>=20){show('최대 20개까지 등록할 수 있습니다.',true);return}subs.push(v);subsDirty=true;renderSubs()}
const safe=v=>String(v??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const when=v=>v?new Date(v*1000).toLocaleString('ko-KR',{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}):'기록 없음';
async function request(options={}){const r=await fetch('/api/anki',{...options,headers:{'Content-Type':'application/json'}});if(!r.ok)throw new Error();return r.json()}
function word(w){const t=safe(w.highlight||""),at=t?safe(w.sentence).indexOf(t):-1,hi=at<0?safe(w.sentence):safe(w.sentence).slice(0,at)+`<mark>${t}</mark>`+safe(w.sentence).slice(at+t.length);
return `<article class="word-card"><div class="word-top"><h3>${safe(w.word)}</h3>${w.cefr?`<span class="cefr c-${safe(w.cefr.toLowerCase())}">${safe(w.cefr)}</span>`:''}</div><p class="meaning">${safe(w.definition)}</p>${w.sentence?`<blockquote>${hi}</blockquote>`:''}${w.academic?`<p class="academic">${safe(w.academic)}</p>`:''}${w.url?`<a class="source" href="${safe(w.url)}" target="_blank" rel="noopener">r/${safe(w.subreddit)} ↗</a>`:''}</article>`}
function render(d){$('connection').textContent='연결됨 · '+new Date(d.updated*1000).toLocaleTimeString('ko-KR');
$('state').textContent=!d.enabled?'자동화 꺼짐':d.error?'점검 필요':'정상';$('last-run').textContent='최근 '+when(d.lastRun);
$('today').textContent=d.addedToday+'개';$('total').textContent=d.totalCards+'개';$('pending').textContent=d.pending+'개';
const lv={};d.words.forEach(w=>{if(w.cefr)lv[w.cefr]=(lv[w.cefr]||0)+1});$('levels').textContent=Object.keys(lv).sort().map(k=>`${k} ${lv[k]}`).join(' · ')||'새 단어';
$('error-banner').hidden=!d.error;$('error-banner').textContent='⚠ '+d.error;
$('word-count').textContent=d.words.length?`${d.words.length}개 · 카드 앞면에 원문 예문, 뒷면에 뜻·학술 예문`:'';
$('words').innerHTML=d.words.length?d.words.map(word).join(''):'<p class="empty">오늘 추가된 단어가 없습니다. 20:30에 수집합니다.</p>';
$('briefing').textContent=d.briefing||'오늘 브리핑이 아직 없습니다.';$('log').textContent=(d.log||[]).join('\n')||'—';
if(!dirty)$('enabled').checked=d.enabled;if(!subsDirty){subs=[...d.subreddits];renderSubs()}}
function show(t,error=false){$('notice').hidden=false;$('notice').className=error?'error':'';$('notice').textContent=t;setTimeout(()=>{$('notice').hidden=true},4000)}
async function refresh(){try{render(await request())}catch{$('connection').textContent='연결 실패'}}
async function save(action){try{render(await request({method:'POST',body:JSON.stringify({enabled:$('enabled').checked,action,subreddits:subs})}));dirty=false;subsDirty=false;show(action==='run_now'?'실행을 시작했습니다. 1~3분 뒤 반영됩니다.':$('enabled').checked?'설정을 저장했습니다.':'자동화를 껐습니다. 21시 브리핑에서도 제외됩니다.')}catch{show('저장하지 못했습니다.',true)}}
$('enabled').oninput=()=>dirty=true;
$('sub-input').onkeydown=e=>{if(e.isComposing)return;if(e.key==='Enter'||e.key===','||e.key===' '){e.preventDefault();addSub(e.target.value);e.target.value=''}else if(e.key==='Backspace'&&!e.target.value&&subs.length>1){subs.pop();subsDirty=true;renderSubs()}};
$('sub-input').onblur=e=>{if(e.target.value.trim()){addSub(e.target.value);e.target.value=''}};
$('tag-box').onclick=e=>{const b=e.target.closest('button[data-i]');if(b){if(subs.length<=1){show('최소 1개는 있어야 합니다.',true);return}subs.splice(+b.dataset.i,1);subsDirty=true;renderSubs()}else if(e.target===e.currentTarget||e.target.id==='tags')$('sub-input').focus()};
$('suggest').onclick=e=>{const b=e.target.closest('button[data-add]');if(b)addSub(b.dataset.add)};$('settings').onsubmit=e=>{e.preventDefault();save('save')};$('run-now').onclick=()=>save('run_now');refresh();setInterval(refresh,15000);
