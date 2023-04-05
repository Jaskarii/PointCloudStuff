#include <iostream>
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/feature.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/common/pca.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/voxel_grid.h>
#include "main.h"

int main (int argc, char** argv)
{

  // Read in point cloud data from file
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltered (new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<int> indices;

  pcl::io::loadPCDFile<pcl::PointXYZ> ("./1665150264219180.pcd", *cloud);

  // Create PCLVisualizer object
  pcl::visualization::PCLVisualizer viewer("Point Cloud Viewer");

  // Set background color to black
  viewer.setBackgroundColor (0, 0, 0);



  pcl::removeNaNFromPointCloud(*cloud, *cloudFiltered, indices);

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> trees = find_trees(cloudFiltered);

  trees = filter_trees_by_height(trees, 3);

  for (size_t i = 0; i < trees.size(); i++)
  {
      rotate_tree_to_z_axis(trees[i]);
      filter_branches(trees[i],0.02,1,5);
      std::string name = "tree ";
      name += std::to_string(i);
      
      calculate_curvature(trees[i]);
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler (trees[i], 0, 255, 0);
      viewer.addPointCloud<pcl::PointXYZ> (trees[i], color_handler, name);
  }

  // Set point size to 2 pixels
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "tree");

  // Spin viewer window
  while (!viewer.wasStopped ())
  {
    viewer.spinOnce (10, false);
  }

  return 0;
}

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> find_trees(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud)
{
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> trees;
  // Perform clustering to find tree clusters
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (input_cloud);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (0.15);
  ec.setMinClusterSize (150);
  ec.setMaxClusterSize (100000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (input_cloud);
  ec.extract (cluster_indices);

  // Extract each tree cluster as a separate point cloud
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
      tree_cloud->points.push_back (input_cloud->points[*pit]);
    tree_cloud->width = tree_cloud->points.size ();
    tree_cloud->height = 1;
    tree_cloud->is_dense = true;

    trees.push_back(tree_cloud);
  }

  return trees;
}

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> filter_trees_by_height(
    const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& trees, float min_height) 
    {
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> filtered_trees;
  for (const auto& tree : trees) {
    // Find the minimum and maximum z-coordinates of the points in the tree
    float min_z = std::numeric_limits<float>::max();
    float max_z = std::numeric_limits<float>::min();
    for (const auto& point : *tree) {
      if (point.z < min_z) min_z = point.z;
      if (point.z > max_z) max_z = point.z;
    }
    float tree_height = max_z - min_z;
    // If the tree's height is above the minimum threshold, add it to the list of filtered trees
    if (tree_height >= min_height) {
      filtered_trees.push_back(tree);
    }
  }
  return filtered_trees;
}

void rotate_tree_to_z_axis(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  int num_top_points = 10;
  // Filter the point cloud using a voxel grid filter to reduce noise
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  vg.setInputCloud(cloud);
  vg.setLeafSize(0.01f, 0.01f, 0.01f); // adjust as necessary
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  vg.filter(*filtered_cloud);

  // Estimate surface normals of the filtered point cloud
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud(filtered_cloud);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
  ne.setSearchMethod(tree);
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  ne.setRadiusSearch(0.1); // adjust as necessary
  ne.compute(*normals);

  // Compute the centroid of the filtered point cloud
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*filtered_cloud, centroid);

  // Extract the top and bottom points of the tree and compute the centerline
  pcl::PointCloud<pcl::PointXYZ>::Ptr top_points(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr bottom_points(new pcl::PointCloud<pcl::PointXYZ>);
  for (int i = 0; i < num_top_points; i++) {
    top_points->push_back(filtered_cloud->points[i]);
  }
  for (int i = filtered_cloud->size() - 1; i >= filtered_cloud->size() - num_top_points; i--) {
    bottom_points->push_back(filtered_cloud->points[i]);
  }
  Eigen::Vector3f centerline = Eigen::Vector3f::Zero();
  for (int i = 0; i < top_points->size(); i++) {
    for (int j = 0; j < bottom_points->size(); j++) {
      Eigen::Vector3f p1(top_points->points[i].x, top_points->points[i].y, top_points->points[i].z);
      Eigen::Vector3f p2(bottom_points->points[j].x, bottom_points->points[j].y, bottom_points->points[j].z);
      centerline += (p1 - p2);
    }
  }
  centerline.normalize();

  // Compute the angle of rotation required to align the centerline with the Z-axis
  Eigen::Vector3f z_axis(0, 0, 1);
  float angle = std::acos(centerline.dot(z_axis) / (centerline.norm() * z_axis.norm()));

  // Compute the axis of rotation
  Eigen::Vector3f axis = centerline.cross(z_axis).normalized();

  // Construct the transformation matrix and apply it to the point cloud
  Eigen::Affine3f transform(Eigen::AngleAxisf(angle, axis));
  Eigen::Matrix4f rotation_matrix = transform.matrix();
  rotation_matrix.block<3, 1>(0, 3) = centroid.head<3>();
  pcl::transformPointCloud(*cloud, *cloud, rotation_matrix);
}

void filter_branches(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double box_width, double box_height, int min_neighbors)
{
  for (size_t i = cloud->size() -1; i > 0; i--)
  {
    pcl::PointXYZ point = (*cloud)[i];
    double xMin = point.x -box_width;
    double xMax = point.x +box_width;
    double yMin = point.y -box_width;
    double yMax = point.y +box_width;
    double zMin = point.z -box_height;
    double zMax = point.z +box_height;
    int count = 0;
    for (size_t j = 0; j < cloud->size(); j++)
    {
      pcl::PointXYZ testPoint = (*cloud)[j];
      if(testPoint.x < xMin || testPoint.x > xMax)
      {
        continue;
      }
      if(testPoint.y < yMin || testPoint.y > yMax)
      {
        continue;
      }
      if(testPoint.z < zMin || testPoint.z > zMax)
      {
        continue;
      }
      count++;
    }

    if(count < min_neighbors)
    {
      cloud->erase(cloud->begin() + i);
    }
    
  }
  
}

void calculate_curvature(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    // Compute surface normals using OpenMP
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setRadiusSearch(1.5);  // Set radius of the sphere for neighborhood search
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.compute(*normals);

    // Compute principal curvatures
    pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> pce;
    pce.setInputCloud(cloud);
    pce.setInputNormals(normals);
    pcl::IndicesPtr indices(new std::vector<int>);
    indices->resize(cloud->size());
    for (int i = 0; i < cloud->size(); i++) {
        (*indices)[i] = i;
    }
    pce.setIndices(indices);
    pce.setRadiusSearch(1.5);  // Set radius of the sphere for neighborhood search
    pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principal_curvatures(new pcl::PointCloud<pcl::PrincipalCurvatures>);
    pce.compute(*principal_curvatures);

    // Compute the mean curvature
    double mean_curvature = 0;
    for (int i = 0; i < cloud->size(); i++) {
        mean_curvature += principal_curvatures->at(i).pc1 + principal_curvatures->at(i).pc2;
    }
    mean_curvature /= cloud->size();

    std::cout << "Mean curvature: " << mean_curvature << std::endl;

}
