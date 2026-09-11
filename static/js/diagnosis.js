/* ==================================================================
   Heart Disease Prediction - diagnosis form behaviour
   Plain JavaScript, no frameworks.

     1. live BMI + WHO category as height / weight change
     2. range checking on every numeric field
     3. block submit when systolic <= diastolic

   IMPORTANT: every check here is repeated on the server in
   ml_engine.validate_form(). Anything done in the browser can be
   bypassed, so this is a convenience for the user, not a security
   control.
   ================================================================== */

document.addEventListener('DOMContentLoaded', function () {

  var form = document.getElementById('diagnosisForm');
  if (!form) return;                       // not on the diagnosis page

  var heightEl = document.getElementById('height');
  var weightEl = document.getElementById('weight');
  var apHiEl   = document.getElementById('ap_hi');
  var apLoEl   = document.getElementById('ap_lo');

  var bmiValue    = document.getElementById('bmiValue');
  var bmiCategory = document.getElementById('bmiCategory');
  var bmiReadout  = document.getElementById('bmiReadout');
  var bpWarning   = document.getElementById('bpWarning');
  var submitBtn   = document.getElementById('submitBtn');

  var COLOURS = {
    blue:  '#3b82f6',
    green: '#10b981',
    amber: '#f59e0b',
    red:   '#ef4444',
    grey:  '#9ca3af'
  };


  /* ---------------------------------------------------------------
     1. LIVE BMI
     Same formula and the same WHO bands used on the server.
     --------------------------------------------------------------- */
  function bmiCategoryFor(bmi) {
    if (bmi < 18.5) return { label: 'Underweight',   colour: COLOURS.blue };
    if (bmi < 25)   return { label: 'Normal weight', colour: COLOURS.green };
    if (bmi < 30)   return { label: 'Overweight',    colour: COLOURS.amber };
    return            { label: 'Obese',         colour: COLOURS.red };
  }

  function updateBMI() {
    var height = parseFloat(heightEl && heightEl.value);
    var weight = parseFloat(weightEl && weightEl.value);

    if (!height || !weight || height <= 0) {
      bmiValue.textContent = '—';
      bmiCategory.textContent = 'enter height and weight';
      bmiCategory.style.color = COLOURS.grey;
      bmiValue.style.color = COLOURS.grey;
      bmiReadout.style.borderLeftColor = COLOURS.grey;
      return;
    }

    var metres = height / 100;
    var bmi = weight / (metres * metres);
    var rounded = Math.round(bmi * 10) / 10;
    var category = bmiCategoryFor(bmi);

    bmiValue.textContent = rounded.toFixed(1);
    bmiValue.style.color = category.colour;
    bmiCategory.textContent = category.label;
    bmiCategory.style.color = category.colour;
    bmiReadout.style.borderLeftColor = category.colour;
  }


  /* ---------------------------------------------------------------
     2. RANGE CHECKING
     Each numeric input carries data-min / data-max / data-label,
     so one function handles them all.
     --------------------------------------------------------------- */
  function showFieldError(input, message) {
    var box = document.querySelector('[data-error-for="' + input.id + '"]');
    if (message) {
      input.classList.add('invalid');
      if (box) { box.textContent = message; box.classList.add('show'); }
    } else {
      input.classList.remove('invalid');
      if (box) { box.textContent = ''; box.classList.remove('show'); }
    }
  }

  function checkRange(input) {
    if (!input) return true;
    var min   = parseFloat(input.getAttribute('data-min'));
    var max   = parseFloat(input.getAttribute('data-max'));
    var label = input.getAttribute('data-label') || input.name;
    var value = parseFloat(input.value);

    if (input.value === '' || isNaN(value)) {
      showFieldError(input, label + ' is required.');
      return false;
    }
    if (value < min || value > max) {
      showFieldError(input, label + ' must be between ' + min + ' and ' + max + '.');
      return false;
    }
    showFieldError(input, null);
    return true;
  }


  /* ---------------------------------------------------------------
     3. BLOOD PRESSURE RELATIONSHIP
     Systolic is always higher than diastolic in a living patient.
     The pulse pressure (the gap) also has to be plausible - the
     models never saw a gap below 20 or above 100.
     --------------------------------------------------------------- */
  function checkBloodPressure() {
    var hi = parseFloat(apHiEl && apHiEl.value);
    var lo = parseFloat(apLoEl && apLoEl.value);

    if (isNaN(hi) || isNaN(lo)) {
      bpWarning.style.display = 'none';
      return true;
    }

    var message = null;

    if (hi <= lo) {
      message = 'Systolic blood pressure (' + hi + ') must be higher than diastolic (' +
                lo + '). The two readings may be swapped.';
    } else {
      var gap = hi - lo;
      if (gap < 20) {
        message = 'Pulse pressure is only ' + gap +
                  ' mmHg. A gap below 20 mmHg is not physiologically plausible.';
      } else if (gap > 100) {
        message = 'Pulse pressure is ' + gap +
                  ' mmHg. A gap above 100 mmHg is not physiologically plausible.';
      }
    }

    if (message) {
      bpWarning.textContent = message;
      bpWarning.style.display = 'block';
      apHiEl.classList.add('invalid');
      apLoEl.classList.add('invalid');
      return false;
    }

    bpWarning.style.display = 'none';
    apHiEl.classList.remove('invalid');
    apLoEl.classList.remove('invalid');
    return true;
  }


  /* ---------------------------------------------------------------
     4. BMI PLAUSIBILITY
     A BMI outside 12-60 means height or weight was almost certainly
     mistyped, so the form is blocked rather than sending nonsense
     to the model.
     --------------------------------------------------------------- */
  function checkBMIPlausible() {
    var height = parseFloat(heightEl && heightEl.value);
    var weight = parseFloat(weightEl && weightEl.value);
    if (!height || !weight) return true;

    var metres = height / 100;
    var bmi = weight / (metres * metres);

    if (bmi < 12 || bmi > 60) {
      showFieldError(weightEl,
        'This height and weight give a BMI of ' + bmi.toFixed(1) +
        ', which is outside the plausible range. Please re-check them.');
      return false;
    }
    // Only clear the message if the range check itself is happy
    if (!weightEl.classList.contains('invalid')) showFieldError(weightEl, null);
    return true;
  }


  /* ---------------------------------------------------------------
     4b. PATIENT NAME AND REFERENCE
     Same rules as ml_engine.validate_form() on the server.
     --------------------------------------------------------------- */
  var nameEl = document.getElementById('patient_name');
  var refEl  = document.getElementById('patient_id');

  function checkName() {
    if (!nameEl) return true;
    var v = nameEl.value.trim();
    if (!v)            return showFieldError(nameEl, 'Patient name is required.') || false;
    if (v.length < 2)  return showFieldError(nameEl, 'At least 2 characters.') || false;
    if (v.length > 60) return showFieldError(nameEl, 'No more than 60 characters.') || false;
    if (!/^[A-Za-z .'\-]+$/.test(v))
      return showFieldError(nameEl, 'Letters, spaces, hyphens and apostrophes only.') || false;
    showFieldError(nameEl, null);
    return true;
  }

  /* Patient reference must be the house format: PT- then 3-12 digits */
  function checkRef() {
    if (!refEl) return true;
    var v = refEl.value.trim().toUpperCase();
    if (!v) { showFieldError(refEl, null); return true; }   // optional
    if (v.indexOf('PT-') !== 0)
      return showFieldError(refEl, "Patient ID must start with 'PT-', e.g. PT-1001.") || false;
    if (!/^PT-\d{3,12}$/.test(v))
      return showFieldError(refEl, "Use 'PT-' followed by 3 to 12 digits, e.g. PT-1001.") || false;
    showFieldError(refEl, null);
    return true;
  }

  if (nameEl) { nameEl.addEventListener('blur', checkName); nameEl.addEventListener('input', function(){ if(nameEl.classList.contains('invalid')) checkName(); }); }
  if (refEl)  { refEl.addEventListener('blur', checkRef);  refEl.addEventListener('input', function(){ if(refEl.classList.contains('invalid')) checkRef(); }); }


  /* ---------------------------------------------------------------
     5. WIRE EVERYTHING UP
     --------------------------------------------------------------- */
  function revalidate() {
    updateBMI();
    checkBloodPressure();
  }

  [heightEl, weightEl].forEach(function (el) {
    if (!el) return;
    el.addEventListener('input', function () { updateBMI(); checkBMIPlausible(); });
    el.addEventListener('blur',  function () { checkRange(el); checkBMIPlausible(); });
  });

  [apHiEl, apLoEl].forEach(function (el) {
    if (!el) return;
    el.addEventListener('input', checkBloodPressure);
    el.addEventListener('blur',  function () { checkRange(el); checkBloodPressure(); });
  });

  var ageEl = document.getElementById('age');
  if (ageEl) ageEl.addEventListener('blur', function () { checkRange(ageEl); });

  // Final gate before the form is sent
  form.addEventListener('submit', function (e) {
    var ok = true;

    ['age', 'height', 'weight', 'ap_hi', 'ap_lo'].forEach(function (id) {
      if (!checkRange(document.getElementById(id))) ok = false;
    });
    if (!checkName()) ok = false;
    if (!checkRef()) ok = false;
    if (!checkBloodPressure()) ok = false;
    if (!checkBMIPlausible()) ok = false;

    if (!ok) {
      e.preventDefault();
      // Bring the first problem into view
      var firstBad = form.querySelector('.invalid');
      if (firstBad) {
        firstBad.scrollIntoView({ behavior: 'smooth', block: 'center' });
        firstBad.focus();
      }
      return false;
    }

    // Stop double submission while the server works
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.textContent = 'Analysing…';
    }
  });

  // Draw the initial state from whatever the form was rendered with
  revalidate();
});
