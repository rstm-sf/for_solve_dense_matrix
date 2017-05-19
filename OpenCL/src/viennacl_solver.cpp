#include "viennacl_solver.h"

int32_t viennacl_solve(const int32_t n) {
	ublas_mat ublas_A(n, n);
	ublas_vec ublas_X(n);
	fill_matrix(ublas_A, 100.0);
	fill_vector(ublas_X, 10.0);

	vcl_mat vcl_A(n, n);
	vcl_vec vcl_X(n);
	viennacl::copy(ublas_A, vcl_A);
	viennacl::copy(ublas_X, vcl_X);

	vcl_vec vcl_B = viennacl::linalg::prod(vcl_A, vcl_X);

	vcl_mat vcl_LU = vcl_A;
	vcl_X = vcl_B;

	viennacl::tools::timer timer;
	double t1 = 0.0, t2 = 0.0, t3 = 0.0;

	printf("\nStart viennacl getrf...\n");
	viennacl::backend::finish();
	timer.start();
	t1 = timer.get();
	viennacl::linalg::lu_factorize(vcl_LU);
	viennacl::backend::finish();
	t1 = timer.get() - t1;
	printf("Stop viennacl getrf...\nTime calc: %f (s.)\n", t1);
	print_to_file_time("viennacl_getrf_time.log", n, t1);

	printf("Start viennacl getrs...\n");
	viennacl::backend::finish();
	t2 = timer.get();
	viennacl::linalg::lu_substitute(vcl_LU, vcl_X);
	viennacl::backend::finish();
	t2 = timer.get() - t2;
	printf("Stop viennacl getrs...\nTime calc: %f (s.)\n", t2);
	print_to_file_time("viennacl_getrs_time.log", n, t2);
	
	const vcl_scalar alpha = (FLOAT)1.0;
	const vcl_scalar beta = (FLOAT)(-1.0);
	printf("Start viennacl gemv...\n");
	viennacl::backend::finish();
	t3 = timer.get();
	vcl_vec vcl_Ax_b = alpha * viennacl::linalg::prod(vcl_A, vcl_X) + beta * vcl_B;
	viennacl::backend::finish();
	t3 = timer.get() - t3;
	printf("Start viennacl gemv...\nTime calc: %f (s.)\n", t3);
	print_to_file_time("viennacl_gemv_time.log", n, t3);

	const FLOAT nrm_b = viennacl::linalg::norm_2(vcl_B);
	assert(("norm_2(b) <= 1e-24!", nrm_b > 1e-24));

	const FLOAT residual = viennacl::linalg::norm_2(vcl_Ax_b);
	printf("Abs. residual: %e\n", residual);
	const FLOAT relat_residual = residual / nrm_b;
	printf("Relative residual: %e\n\n", relat_residual);
	print_to_file_residual("viennacl_relat_residual.log", n, relat_residual);

	return 0;
}

void viennacl_info() {
	typedef std::vector<viennacl::ocl::platform> platforms_type;
	typedef std::vector<viennacl::ocl::device> devices_type;

	platforms_type platforms = viennacl::ocl::get_platforms();
	bool is_first_element = true;

	for (platforms_type::iterator platform_iter  = platforms.begin();
	     platform_iter != platforms.end();
	     ++platform_iter)
	{
		devices_type devices = platform_iter->devices(CL_DEVICE_TYPE_ALL);
		std::cout << "# =========================================" << std::endl;
		std::cout << "#         Platform Information             " << std::endl;
		std::cout << "# =========================================" << std::endl;
		std::cout << "#" << std::endl;
		std::cout << "# Vendor and version: " << platform_iter->info() << std::endl;
		std::cout << "#" << std::endl;

		if (is_first_element) {
			std::cout << "# ViennaCL uses this OpenCL platform by default." << std::endl;
			is_first_element = false;
		}

		std::cout << "# " << std::endl;
		std::cout << "# Available Devices: " << std::endl;
		std::cout << "# " << std::endl;

		for (devices_type::iterator iter = devices.begin(); iter != devices.end(); ++iter) {
			std::cout << std::endl;
			std::cout << "  -----------------------------------------" << std::endl;
			std::cout << iter->full_info();
			std::cout << "ViennaCL Device Architecture:  " << iter->architecture_family() << std::endl;
			std::cout << "ViennaCL Database Mapped Name: ";
			std::cout << viennacl::device_specific::builtin_database::get_mapped_device_name(
			                                          iter->name(), iter->vendor_id()) << std::endl;
			std::cout << "  -----------------------------------------" << std::endl;
		}

		std::cout << std::endl;
		std::cout << "###########################################" << std::endl;
		std::cout << std::endl;
	}
}