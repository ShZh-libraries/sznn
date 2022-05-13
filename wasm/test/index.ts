import "./public/init";

// Wait for initializaiton done
before((done) => {
    setTimeout(done, 500);  // Wait for 500 miliseconds
})

import "./binaryop.spec";
import "./concat.spec";
import "./conv.spec";
import "./unaryop.spec";