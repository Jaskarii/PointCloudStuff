#include <iostream>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>

/// @brief Separate trees from the point cloud data
/// @param input_cloud 
/// @return 
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> find_trees(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud);

///Return list of trees in from the point clouds that where above min height
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> filter_trees_by_height(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& trees, float min_height);

///Rotate given point cloud so that it points to Z axis
void rotate_tree_to_z_axis(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

//Filters Branches out of trees
void filter_branches(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double box_width, double box_height, int min_neighbors);
