import "./public/init";

import chai from "chai";
const chaiDeepCloseTo = require("chai-deep-closeto");

chai.use(chaiDeepCloseTo);

// Wait for initializaiton done
before((done) => {
  setTimeout(done, 500); // Wait for 500 miliseconds
});

const expect = chai.expect;
export { expect };

import "./binaryop.spec";
import "./concat.spec";
import "./conv.spec";
import "./instancenorm.spec";
import "./padding.spec";
import "./pooling.spec";
import "./relu.spec";
import "./unaryop.spec";
import "./upsample.spec";
