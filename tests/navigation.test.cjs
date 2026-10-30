const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const context = vm.createContext({});
vm.runInContext(fs.readFileSync('web/navigation.js', 'utf8') + '\nthis.product = {LAB_MODULES, COURSE_PATHS, resolveRoute};', context);
const {LAB_MODULES, COURSE_PATHS, resolveRoute} = context.product;
const plain = value => JSON.parse(JSON.stringify(value));

test('only the four conceptual modules occupy the first level', () => {
  assert.deepEqual(Object.keys(LAB_MODULES), ['common', 'search', 'ads', 'recommendation']);
  const html = fs.readFileSync('web/index.html', 'utf8');
  const primary = html.match(/<nav id="module-nav"[\s\S]*?<\/nav>/)[0];
  assert.equal((primary.match(/data-route=/g) || []).length, 4);
  assert.ok(!primary.includes('two-tower'));
});
test('old deep links preserve their tool and acquire the correct parent', () => {
  assert.deepEqual(plain(resolveRoute('#experiments')), {module:'recommendation', page:'two-tower'});
  assert.deepEqual(plain(resolveRoute('#search')), {module:'search', page:'retrieval'});
  assert.deepEqual(plain(resolveRoute('#ads')), {module:'ads', page:'prediction'});
  assert.deepEqual(plain(resolveRoute('#research')), {module:'common', page:'records'});
});
test('every module page has a real panel and survives route reload', () => {
  const html = fs.readFileSync('web/index.html', 'utf8');
  for (const [module, config] of Object.entries(LAB_MODULES)) {
    for (const [page, , panel] of config.pages) {
      assert.ok(html.includes(`<section id="${panel}"`), `${module}/${page} missing panel`);
      assert.deepEqual(plain(resolveRoute(`#${module}/${page}`)), {module, page});
    }
  }
  assert.deepEqual(plain(resolveRoute('#search/two-tower')), {module:'search',page:'path'});
  assert.deepEqual(plain(resolveRoute('#unknown')), {module:'common',page:'start'});
});
test('all existing notebooks belong to exactly one domain; rec basics remain rec', () => {
  const all = Object.values(COURSE_PATHS).flatMap(phases => phases.flatMap(p => plain(p.chapters)));
  assert.equal(all.length, 35);
  assert.deepEqual([...all].sort((a,b)=>a-b), Array.from({length:35},(_,i)=>i));
  assert.equal(COURSE_PATHS.search.flatMap(p=>p.chapters).length, 7);
  assert.equal(COURSE_PATHS.ads.flatMap(p=>p.chapters).length, 7);
  const rec = COURSE_PATHS.recommendation.flatMap(p=>p.chapters);
  for (const n of [0,1,2,3]) assert.ok(rec.includes(n));
  assert.deepEqual(plain(COURSE_PATHS.common.flatMap(p=>p.chapters)), [31,32,33,34]);
  assert.equal(rec.length, 17);
});
